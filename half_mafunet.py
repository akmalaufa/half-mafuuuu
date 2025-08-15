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


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, act=nn.ReLU(inplace=True)):
        super().__init__()
        if p is None: p = k // 2
        self.dw = ConvBNAct(in_c, in_c, k, s, p, g=in_c, act=act)
        self.pw = ConvBNAct(in_c, out_c, 1, 1, 0, act=act)
    def forward(self, x): return self.pw(self.dw(x))


# Dual-gate fusers
class SDGF(nn.Module):
    """Spatial Dual-Gate Fusion: fuse two spatial gates then apply."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable mix
    def forward(self, x, g1, g2):
        g = torch.clamp(self.alpha, 0, 1) * g1 + torch.clamp(1 - self.alpha, 0, 1) * g2
        return x * g


class CDGF(nn.Module):
    """Channel Dual-Gate Fusion: fuse two channel gates then apply."""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.proj  = nn.Conv2d(channels, channels, 1, 1, 0)  # light calibration
        self.sig   = nn.Sigmoid()
    def forward(self, x, g1, g2):
        g = torch.clamp(self.alpha, 0, 1) * g1 + torch.clamp(1 - self.alpha, 0, 1) * g2
        g = self.sig(self.proj(g))
        return x * g


# ASA: Attention Spatial (dual path: conv & dilated conv)
class ASA(nn.Module):
    def __init__(self, k=7, dilation=2):
        super().__init__()
        p = k // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, k, padding=p, bias=False),
            nn.Sigmoid()
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(2, 1, k, padding=dilation * p, dilation=dilation, bias=False),
            nn.Sigmoid()
        )
        self.fuse = SDGF()

    def forward(self, x):
        mean_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        s = torch.cat([mean_map, max_map], dim=1)  # B x 2 x H x W
        g1 = self.conv(s)
        g2 = self.dilated(s)
        return self.fuse(x, g1, g2)


# ACA: Attention Channel (dual pooling scales)
class ACA(nn.Module):
    """
    Channel attention using two avg-pool scales:
      AAP(1): 1x1 global, AAP(2): 2x2 context
    Both -> small MLP (shared width) -> gates -> CDGF.
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

        self.sig = nn.Sigmoid()
        self.fuse = CDGF(channels)

    def forward(self, x):
        g1 = self.sig(self.mlp1(self.pool1(x)))  # B x C x 1 x 1
        p2 = self.pool2(x)                       # B x C x 2 x 2
        g2 = self.sig(self.mlp2(F.adaptive_avg_pool2d(p2, 1)))  # reduce to 1x1
        return self.fuse(x, g1, g2)


# HAM (lightweight) used inside MAF
class HAM(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.local = DepthwiseSeparable(channels, channels, k=3, act=nn.SiLU(inplace=True))
        self.aca   = ACA(channels, r=8)
        self.asa   = ASA()
        self.proj  = ConvBNAct(channels, channels, k=1, act=nn.Identity())

    def forward(self, x):
        y = self.local(x)
        y = self.aca(y)
        y = self.asa(y)
        y = self.proj(y)
        return x + y


# MAF block (stack of HAMs)
class MAF(nn.Module):
    def __init__(self, channels, depth=6):
        super().__init__()
        self.blocks = nn.Sequential(*[HAM(channels) for _ in range(depth)])
    def forward(self, x): return self.blocks(x)


# Encoder/Decoder pieces
class Down(nn.Module):
    """
    Diagram: MaxPool -> MAF -> Conv2d(k=1) (to expand channels)
    """
    def __init__(self, in_c, out_c, maf_depth=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.proj_in = ConvBNAct(in_c, out_c, k=1)  # harmonize first
        self.maf = MAF(out_c, depth=maf_depth)
        self.proj_out = ConvBNAct(out_c, out_c, k=1)
    def forward(self, x):
        x = self.pool(x)
        x = self.proj_in(x)
        x = self.maf(x)
        x = self.proj_out(x)
        return x



# Bridge Attention Column: ACA -> ASA on encoder features
class BridgeAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.aca = ACA(channels)
        self.asa = ASA()
    def forward(self, x):
        return self.asa(self.aca(x))


# MAFU-Net
class MAFUNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, base_c=16, maf_depth=2):
        super().__init__()
        C = base_c

        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, C, 3),
            ConvBNAct(C, C, 3)
        )

        # Encoder channels
        self.enc1 = Down(C,   2*C, maf_depth)  # 1/2
        self.enc2 = Down(2*C, 4*C, maf_depth)  # 1/4
        self.enc3 = Down(4*C, 8*C, maf_depth)  # 1/8
        self.enc4 = Down(8*C,16*C, maf_depth)  # 1/16
        self.enc5 = Down(16*C,32*C, maf_depth) # 1/32 (bottom)

        # Bridge attention for each encoder output (to be fused later)
        self.bridge0 = BridgeAttention(C)       # stem
        self.bridge1 = BridgeAttention(2*C)
        self.bridge2 = BridgeAttention(4*C)
        self.bridge3 = BridgeAttention(8*C)
        self.bridge4 = BridgeAttention(16*C)
        self.bridge5 = BridgeAttention(32*C)

        # Lateral projections to align channels before fusion (all -> C)
        self.lat0 = ConvBNAct(C,     C, 1)
        self.lat1 = ConvBNAct(2*C,   C, 1)
        self.lat2 = ConvBNAct(4*C,   C, 1)
        self.lat3 = ConvBNAct(8*C,   C, 1)
        self.lat4 = ConvBNAct(16*C,  C, 1)
        self.lat5 = ConvBNAct(32*C,  C, 1)

        # Fusion head (concat 6 scales -> mix -> C)
        self.fuse = nn.Sequential(
            ConvBNAct(6*C, 2*C, 3),
            ConvBNAct(2*C,   C, 3)
        )

        # Prediction head
        self.head = nn.Conv2d(C, out_ch, kernel_size=1)

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
    print("Output:", y.shape)      # -> (2, 1, 256, 256)
