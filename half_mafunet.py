import torch
import torch.nn as nn
import torch.nn.functional as F

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


# Enhanced MAFU-Net with large kernels throughout for better performance
class MAFUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_c=16, maf_depth=2, dropout_rate=0.1):
        super().__init__()
        C = base_c
        self.dropout_rate = dropout_rate

        # Enhanced stem with large kernels
        self.stem = nn.Sequential(
            LargeKernelConv(in_ch, C, 7),  # Large kernel 7x7
            LargeKernelConv(C, C, 7),      # Large kernel 7x7
            nn.Dropout2d(dropout_rate)
        )

        # Encoder channels
        self.enc1 = Down(C,   2*C, maf_depth)  # 1/2
        self.enc2 = Down(2*C, 4*C, maf_depth)  # 1/4
        self.enc3 = Down(4*C, 8*C, maf_depth)  # 1/8
        self.enc4 = Down(8*C,16*C, maf_depth)  # 1/16
        self.enc5 = Down(16*C,32*C, maf_depth) # 1/32 (bottom)

        # Enhanced bridge attention with large kernels
        self.bridge0 = BridgeAttention(C)       # stem
        self.bridge1 = BridgeAttention(2*C)
        self.bridge2 = BridgeAttention(4*C)
        self.bridge3 = BridgeAttention(8*C)
        self.bridge4 = BridgeAttention(16*C)
        self.bridge5 = BridgeAttention(32*C)

        # Lateral projections with large kernels to align channels before fusion (all -> C)
        self.lat0 = LargeKernelConv(C,     C, 7)
        self.lat1 = LargeKernelConv(2*C,   C, 7)
        self.lat2 = LargeKernelConv(4*C,   C, 7)
        self.lat3 = LargeKernelConv(8*C,   C, 7)
        self.lat4 = LargeKernelConv(16*C,  C, 7)
        self.lat5 = LargeKernelConv(32*C,  C, 7)

        # Enhanced fusion head with large kernels
        self.fuse = nn.Sequential(
            LargeKernelConv(6*C, 2*C, 7),  # Large kernel fusion
            nn.Dropout2d(dropout_rate),
            LargeKernelConv(2*C,   C, 7),   # Large kernel refinement
            nn.Dropout2d(dropout_rate)
        )

        # Enhanced prediction head with large kernel
        self.head = nn.Sequential(
            LargeKernelConv(C, C, 7, act=nn.SiLU(inplace=True)),
            nn.Conv2d(C, out_ch, kernel_size=1)
        )

    def forward(self, x):
        B, _, H, W = x.shape

        # Stem and encoder
        x0 = self.stem(x)          # B x C x H x W
        s0 = self.bridge0(x0)      # attentioned skip at HxW

        x1 = self.enc1(x0)         # B x 2C x H/2
        s1 = self.bridge1(x1)

        x2 = self.enc2(x1)         # B x 4C x H/4
        s2 = self.bridge2(x2)

        x3 = self.enc3(x2)         # B x 8C x H/8
        s3 = self.bridge3(x3)

        x4 = self.enc4(x3)         # B x 16C x H/16
        s4 = self.bridge4(x4)

        x5 = self.enc5(x4)         # B x 32C x H/32
        s5 = self.bridge5(x5)

        # Align channels and upsample all to HxW (simple top-down fusion)
        f0 = self.lat0(s0)
        f1 = F.interpolate(self.lat1(s1), size=(H, W), mode='bilinear', align_corners=True)
        f2 = F.interpolate(self.lat2(s2), size=(H, W), mode='bilinear', align_corners=True)
        f3 = F.interpolate(self.lat3(s3), size=(H, W), mode='bilinear', align_corners=True)
        f4 = F.interpolate(self.lat4(s4), size=(H, W), mode='bilinear', align_corners=True)
        f5 = F.interpolate(self.lat5(s5), size=(H, W), mode='bilinear', align_corners=True)

        # Concatenate and fuse
        fused = torch.cat([f0, f1, f2, f3, f4, f5], dim=1)  # B x 6C x H x W
        fused = self.fuse(fused)                            # B x C x H x W

        logits = self.head(fused)
        return logits


# Quick test
if __name__ == "__main__":
    model = MAFUNet(in_ch=3, out_ch=1, base_c=16, maf_depth=2)
    x = torch.randn(1, 3, 288, 384)
    with torch.no_grad():
        y = model(x)
    print("Output:", y.shape)      # -> (1, 1, 288, 384)
