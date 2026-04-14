import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLevelRoutingAttention(nn.Module):
    """
    Bi-Level Routing Attention (BRA) simplified for YOLO integration.
    Source: BiFormer: Vision Transformer with Bi-Level Routing Attention.
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, 
                 topk=4, side_dwconv=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk_scale = qk_scale or (dim // num_heads) ** -0.5
        self.topk = topk
        self.n_win = n_win

        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.k = nn.Conv2d(dim, dim, kernel_size=1)
        self.v = nn.Conv2d(dim, dim, kernel_size=1)
        
        if side_dwconv > 0:
            self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim)
        else:
            self.lepe = None
            
        self.re_pro = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, C // self.num_heads, -1) # B, h, c, N
        k = k.view(B, self.num_heads, C // self.num_heads, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, -1)
        
        # Attention
        attn = (q.transpose(-2, -1) @ k) * self.qk_scale # B, h, N, N
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1) # B, h, c, N
        out = out.reshape(B, C, H, W)
        
        if self.lepe is not None:
            out = out + self.lepe(x)
            
        out = self.re_pro(out)
        return out + shortcut

class C3k2_BiFormer(nn.Module):
    """
    YOLO11 C3k2 variant with BiFormer (Bi-Level Routing Attention).
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules import Conv, C3k, Bottleneck
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g), BiLevelRoutingAttention(self.c)) 
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)
    model = C3k2_BiFormer(64, 64)
    print(f"C3k2_BiFormer Output: {model(x).shape}")
