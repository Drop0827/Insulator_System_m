import torch
import torch.nn as nn
from ultralytics.nn.modules import C3

__all__ = ['PConv', 'C2f_PConv', 'C3k2_PConv']


# ──────────────────────────────────────────────────────────────────────────────
# PConv: 部分卷积（Partial Convolution）
# 论文来源：FasterNet（CVPR 2023）
# 核心思想：仅对输入的前 1/n_div 通道进行 3×3 卷积，其余通道直接 Identity 传递，
# 以此大幅降低冗余计算量，同时保留特征提取能力。
# ──────────────────────────────────────────────────────────────────────────────
class PConv(nn.Module):
    """
    部分卷积层（Partial Convolution）。
    只对前 dim // n_div 个通道施加 3×3 卷积，其余通道保持不变，
    从而显著减少 FLOPs 并提升推理速度。

    Args:
        dim (int): 输入 / 输出通道数。
        n_div (int): 通道分割比例，默认 4（即 1/4 的通道参与卷积）。
    """

    def __init__(self, dim: int, n_div: int = 4):
        super().__init__()
        # 参与卷积的通道数（前 1/n_div）
        self.dim_conv = dim // n_div
        # 直接传递（Identity）的通道数
        self.dim_untouched = dim - self.dim_conv

        # NOTE: bias=False 是为了与 BatchNorm 配合，避免引入偏置冗余
        self.partial_conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将通道维度切分为「参与卷积」和「直接传递」两部分
        x_conv, x_pass = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x_conv = self.partial_conv(x_conv)
        # 拼回原始通道顺序，保持输入输出 shape 一致
        return torch.cat((x_conv, x_pass), dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数：与 ultralytics 原生 autopad 保持一致
# ──────────────────────────────────────────────────────────────────────────────
def _autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class _Conv(nn.Module):
    """局部 Conv+BN+SiLU，仅供本文件内部使用，避免循环导入。"""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ──────────────────────────────────────────────────────────────────────────────
# C2f_PConv: 以 PConv 为 Bottleneck 的 C2f 变体
# NOTE: C2f_PConv 直接将 PConv 作为特征混合单元，比原 C2f 参数更少、速度更快
# ──────────────────────────────────────────────────────────────────────────────
class C2f_PConv(nn.Module):
    """C2f 变体，将内部 Bottleneck 替换为 PConv，减少冗余计算。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)          # 隐层通道数
        self.cv1 = _Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = _Conv((2 + n) * self.c, c2, 1)
        # NOTE: 每个 PConv 模块的输入通道数等于隐层通道数
        self.m = nn.ModuleList(PConv(self.c) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ──────────────────────────────────────────────────────────────────────────────
# C3k_PConv: 内部使用 PConv 的 C3k 子模块（供 C3k2_PConv 按需调用）
# ──────────────────────────────────────────────────────────────────────────────
class _C3k_PConv(C3):
    """C3k 变体，将内部 Bottleneck 替换为 PConv 序列。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # 用 PConv 序列替代原始 Bottleneck
        self.m = nn.Sequential(*(PConv(c_) for _ in range(n)))


# ──────────────────────────────────────────────────────────────────────────────
# C3k2_PConv: YOLO11 Neck 中的核心改进模块（P-C3k2）
# 与基线 C3k2 的唯一区别：将轻量 Bottleneck 替换为 PConv，
# 在不显著增加参数的前提下提升速度（FPS ↑），并保留足够的特征表达能力。
# ──────────────────────────────────────────────────────────────────────────────
class C3k2_PConv(C2f_PConv):
    """
    P-C3k2：将 PConv 集成到 C3k2 架构的改进模块。
    对应论文「基于改进 YOLOv11 的钢轨缺陷检测实验研究」中的 P-C3k2 模块。

    当 c3k=True 时，使用 _C3k_PConv（深层卷积变体）；
    当 c3k=False 时，直接使用轻量 PConv，速度更快。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False,
                 e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, shortcut, g, e)
        # HACK: 根据 c3k 标志决定使用深层或轻量变体
        # c3k=True 对应 Neck 中特征通道较多的层，False 对应轻量检测头前的融合层
        self.m = nn.ModuleList(
            _C3k_PConv(self.c, self.c, 2, shortcut, g)
            if c3k
            else PConv(self.c)
            for _ in range(n)
        )


# ──────────────────────────────────────────────────────────────────────────────
# 快速验证：输入 BCHW → 输出 BCHW（通道、尺寸不变）
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    batch = torch.randn(1, 256, 40, 40)

    pconv = PConv(dim=256)
    print(f'[PConv]      input: {batch.shape}  output: {pconv(batch).shape}')

    c2f_p = C2f_PConv(256, 256, n=2)
    print(f'[C2f_PConv]  input: {batch.shape}  output: {c2f_p(batch).shape}')

    c3k2_p = C3k2_PConv(256, 256, n=2, c3k=False)
    print(f'[C3k2_PConv] input: {batch.shape}  output: {c3k2_p(batch).shape}')
