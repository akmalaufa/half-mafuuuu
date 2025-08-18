import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import timm
except Exception:
    timm = None

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, g=1, act=nn.ReLU(inplace=True)):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = act
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class LargeKernelConv(nn.Module):
    """Large kernel convolution with multi-scale context"""
    def __init__(self, in_c, out_c, k=7, s=1, p=None, act=nn.ReLU(inplace=True)):
        super().__init__()
        if p is None: p = k // 2
        # Multi-scale large kernels for better spatial context
        self.conv7 = nn.Conv2d(in_c, out_c//2, 7, s, 3, bias=False)
        self.conv9 = nn.Conv2d(in_c, out_c//2, 9, s, 4, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = act
        
    def forward(self, x):
        x1 = self.conv7(x)
        x2 = self.conv9(x)
        x = torch.cat([x1, x2], dim=1)
        return self.act(self.bn(x))


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, act=nn.ReLU(inplace=True)):
        super().__init__()
        if p is None: p = k // 2
        self.dw = ConvBNAct(in_c, in_c, k, s, p, g=in_c, act=act)
        self.pw = ConvBNAct(in_c, out_c, 1, 1, 0, act=act)
    def forward(self, x): return self.pw(self.dw(x))


class LargeKernelDepthwiseSeparable(nn.Module):
    """Large kernel depthwise separable convolution"""
    def __init__(self, in_c, out_c, k=7, s=1, p=None, act=nn.ReLU(inplace=True)):
        super().__init__()
        if p is None: p = k // 2
        self.dw = ConvBNAct(in_c, in_c, k, s, p, g=in_c, act=act)
        self.pw = ConvBNAct(in_c, out_c, 1, 1, 0, act=act)
    def forward(self, x): return self.pw(self.dw(x))


# Enhanced Dual-gate fusers with adaptive weighting
class SDGF(nn.Module):
    """Spatial Dual-Gate Fusion: fuse two spatial gates then apply."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable mix
        self.gamma = nn.Parameter(torch.tensor(1.0))  # adaptive scaling
    def forward(self, x, g1, g2):
        g = torch.clamp(self.alpha, 0, 1) * g1 + torch.clamp(1 - self.alpha, 0, 1) * g2
        return x * torch.sigmoid(self.gamma * g)


class CDGF(nn.Module):
    """Channel Dual-Gate Fusion: fuse two channel gates then apply."""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.proj  = nn.Conv2d(channels, channels, 1, 1, 0)  # light calibration
        self.sig   = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.tensor(1.0))  # adaptive scaling
    def forward(self, x, g1, g2):
        g = torch.clamp(self.alpha, 0, 1) * g1 + torch.clamp(1 - self.alpha, 0, 1) * g2
        g = self.sig(self.proj(g))
        return x * torch.sigmoid(self.gamma * g)


# Enhanced ASA: Attention Spatial with multi-scale context and large kernels
class ASA(nn.Module):
    def __init__(self, k=7, dilation=2):
        super().__init__()
        p = k // 2
        # Multi-scale spatial attention with large kernels
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),  # Large kernel 7x7
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2, 1, 5, padding=2, bias=False),  # Medium kernel 5x5
            nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2, 1, 9, padding=4, bias=False),  # Extra large kernel 9x9
            nn.Sigmoid()
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=4, dilation=2, bias=False),  # Dilated large kernel
            nn.Sigmoid()
        )
        self.fuse = SDGF()
        self.final_fuse = nn.Conv2d(4, 1, 1, bias=False)  # 4 attention maps

    def forward(self, x):
        mean_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        s = torch.cat([mean_map, max_map], dim=1)  # B x 2 x H x W
        
        g1 = self.conv1(s)  # 7x7
        g2 = self.conv2(s)  # 5x5
        g3 = self.conv3(s)  # 9x9
        g4 = self.dilated(s)  # Dilated 7x7
        
        # Multi-scale fusion with large kernels
        gates = torch.cat([g1, g2, g3, g4], dim=1)
        final_gate = torch.sigmoid(self.final_fuse(gates))
        
        return x * final_gate


# Enhanced ACA: Attention Channel with better feature extraction and large kernels
class ACA(nn.Module):
    """
    Channel attention using multiple scales with large kernels:
      AAP(1): 1x1 global, AAP(2): 2x2 context, AAP(3): 3x3 context
    All -> small MLP (shared width) -> gates -> CDGF.
    """
    def __init__(self, channels, r=16):
        super().__init__()
        hidden = max(channels // r, 4)

        # AAP(1): 1x1
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        # AAP(2): 2x2 (flatten via 1x1 conv after pooling)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        # AAP(3): 3x3 context
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.mlp3 = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        # Add max pooling branch for better feature representation
        self.pool4 = nn.AdaptiveMaxPool2d(1)
        self.mlp4 = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        self.sig = nn.Sigmoid()
        self.fuse = CDGF(channels)

    def forward(self, x):
        g1 = self.sig(self.mlp1(self.pool1(x)))  # B x C x 1 x 1
        p2 = self.pool2(x)                       # B x C x 2 x 2
        g2 = self.sig(self.mlp2(F.adaptive_avg_pool2d(p2, 1)))  # reduce to 1x1
        p3 = self.pool3(x)                       # B x C x 3 x 3
        g3 = self.sig(self.mlp3(F.adaptive_avg_pool2d(p3, 1)))  # reduce to 1x1
        g4 = self.sig(self.mlp4(self.pool4(x)))  # B x C x 1 x 1
        
        # Enhanced fusion of four attention maps
        g_combined = (g1 + g2 + g3 + g4) / 4.0
        return self.fuse(x, g1, g_combined)


# Enhanced HAM with better feature processing and large kernels
class HAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Use large kernel depthwise separable for better spatial context
        self.local = LargeKernelDepthwiseSeparable(channels, channels, k=7, act=nn.SiLU(inplace=True))
        self.aca   = ACA(channels, r=8)
        self.asa   = ASA()
        # Large kernel projection
        self.proj  = LargeKernelConv(channels, channels, k=7, act=nn.Identity())
        # Add residual connection with learnable weight
        self.res_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        identity = x
        y = self.local(x)
        y = self.aca(y)
        y = self.asa(y)
        y = self.proj(y)
        return identity + self.res_weight * y


# Enhanced MAF block with better depth utilization and large kernels
class MAF(nn.Module):
    def __init__(self, channels, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([HAM(channels) for _ in range(depth)])
        # Add cross-connections between HAM blocks with large kernel
        if depth > 1:
            self.cross_conv = LargeKernelConv(channels, channels, k=7, act=nn.Identity())
        else:
            self.cross_conv = None
            
    def forward(self, x):
        if self.cross_conv is None:
            return self.blocks(x)
        
        # Enhanced forward with cross-connections
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            outputs.append(x)
        
        # Cross-connection: combine all intermediate outputs
        if len(outputs) > 1:
            cross_feat = torch.stack(outputs, dim=0).mean(dim=0)
            x = x + self.cross_conv(cross_feat)
        
        return x


# Enhanced Down block with better feature propagation and large kernels
class Down(nn.Module):
    """
    Diagram: MaxPool -> MAF -> LargeKernelConv (to expand channels)
    """
    def __init__(self, in_c, out_c, maf_depth=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        # Use large kernel for better feature extraction
        self.proj_in = LargeKernelConv(in_c, out_c, k=7)
        self.maf = MAF(out_c, depth=maf_depth)
        self.proj_out = LargeKernelConv(out_c, out_c, k=7)
        # Add skip connection projection with large kernel
        self.skip_proj = LargeKernelConv(in_c, out_c, k=7) if in_c != out_c else nn.Identity()
        
    def forward(self, x):
        skip = self.skip_proj(x)
        x = self.pool(x)
        x = self.proj_in(x)
        x = self.maf(x)
        x = self.proj_out(x)
        # Add residual connection from input
        return x + F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)


# Enhanced Bridge Attention with large kernel refinement
class BridgeAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.aca = ACA(channels)
        self.asa = ASA()
        # Add large kernel refinement
        self.refine = LargeKernelConv(channels, channels, k=7, act=nn.SiLU(inplace=True))
        
    def forward(self, x):
        x = self.asa(self.aca(x))
        return self.refine(x)


#############################
# Lightweight building blocks
#############################

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, kernel_size=1, dw_kernel_size=3, stride=1, relu=True):
        super().__init__()
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2 if kernel_size > 1 else 0, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=True):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LightweightSE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.avg(x)
        w = self.fc(w)
        return x * w


class GhostDSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.ghost = GhostModule(out_channels, out_channels, ratio=2, kernel_size=1, dw_kernel_size=3, relu=True)
        self.ds = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, act=True)
        self.se = LightweightSE(out_channels, reduction=8)
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.reduce(x)
        x = self.bn0(x)
        x = F.silu(x, inplace=True)
        x = self.ghost(x)
        x = self.ds(x)
        x = self.se(x)
        x = self.drop(x)
        return x


class FPNDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels=16, dropout=0.1):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1, 1, 0, bias=False) for c in in_channels_list])
        self.blocks = nn.ModuleList([GhostDSBlock(out_channels, out_channels, dropout=dropout) for _ in in_channels_list])

    def forward(self, feats):
        # feats: list of feature maps high->low spatial resolution order [C4, C3, C2, C1]
        lat = [l(f) for l, f in zip(self.laterals, feats)]
        x = lat[0]
        outputs = [self.blocks[0](x)]
        for i in range(1, len(lat)):
            x = F.interpolate(x, size=lat[i].shape[2:], mode='bilinear', align_corners=True) + lat[i]
            x = self.blocks[i](x)
            outputs.append(x)
        # return finest scale
        return outputs[-1]


class MAFUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_c=16, maf_depth=2, dropout_rate=0.1, backbone_name: str = 'mobilenetv3_small_050'):
        super().__init__()
        assert timm is not None, "timm is required for MobileNetV3 backbone. Please install timm."
        # Pretrained MobileNetV3-Small backbone; rely on default feature outputs
        # Robust creation: try requested pretrained; fallback to default pretrained; finally no-pretrained
        enc = None
        try:
            enc = timm.create_model(backbone_name, pretrained=True, features_only=True)
        except Exception:
            try:
                enc = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
            except Exception:
                enc = timm.create_model(backbone_name, pretrained=False, features_only=True)
        self.encoder = enc
        enc_channels = self.encoder.feature_info.channels()
        if len(enc_channels) < 4:
            raise RuntimeError(f"Backbone {backbone_name} must provide at least 4 feature maps, got {len(enc_channels)}")

        # Take last 4 stages, ordered deepest->shallowest for FPN
        in_list = enc_channels[-1:-5:-1]
        # Build lightweight FPN decoder with small channel width to fit param budget
        self.decoder_out_channels = base_c  # keep tiny to meet param budget
        self.decoder = FPNDecoder(in_list, out_channels=self.decoder_out_channels, dropout=dropout_rate)

        # Final prediction head
        self.head = nn.Sequential(
            GhostDSBlock(self.decoder_out_channels, self.decoder_out_channels, dropout=dropout_rate),
            nn.Conv2d(self.decoder_out_channels, out_ch, kernel_size=1)
        )

    def forward(self, x):
        H, W = x.shape[2:]
        feats = self.encoder(x)
        # Arrange as [C4, C3, C2, C1] deepest->shallowest
        feats = feats[-1:-5:-1]
        x = self.decoder(feats)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.head(x)
        return x

    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


# Quick test
if __name__ == "__main__":
    model = MAFUNet(in_ch=3, out_ch=1, base_c=16, maf_depth=2)
    x = torch.randn(1, 3, 288, 384)
    with torch.no_grad():
        y = model(x)
    print("Output:", y.shape)      # -> (1, 1, 288, 384)
