import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Optional
from torch import Tensor

import warnings


# ======================================距离+方向位置编码 BEGIN======================================

def shift_scale_points(points, src_range):
    """将点归一化到 [0,1] 区间"""
    min_val, max_val = src_range
    return (points - min_val) / (max_val - min_val + 1e-8)


def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    
    return sq_distances

class DistanceDirectionSineEncoding(nn.Module):
    def __init__(self, num_channels=64, temperature=10000, normalize=False, scale=None, dist_mode="euclidean"):
        """
        :param num_channels: 编码总通道数
        :param temperature: 正余弦频率缩放
        :param normalize: 是否归一化输入
        :param scale: 缩放比例
        :param dist_mode: 距离计算方式 ("euclidean" | "sq_euclidean" | "angular")
        """
        super().__init__()
        self.num_channels = num_channels
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.dist_mode = dist_mode

    def compute_distance(self, p1, p2):
        diff = p2 - p1
        if self.dist_mode == "euclidean":
            return torch.norm(diff, dim=-1, keepdim=True)
        elif self.dist_mode == "sq_euclidean":
            return torch.sum(diff ** 2, dim=-1, keepdim=True)
        elif self.dist_mode == "angular":
            p1_norm = p1 / (torch.norm(p1, dim=-1, keepdim=True) + 1e-8)
            p2_norm = p2 / (torch.norm(p2, dim=-1, keepdim=True) + 1e-8)
            cos_theta = torch.clamp(torch.sum(p1_norm * p2_norm, dim=-1, keepdim=True), -1.0, 1.0)
            return torch.acos(cos_theta)
        else:
            raise ValueError(f"Unknown distance mode: {self.dist_mode}")

    def get_sine_embeddings(self, coords, num_channels, input_range=None):
        """
        coords: [B, N, N, D] 或者 [B, N, D]
        返回: [B, N, N, num_channels] 或 [B, N, num_channels]
        """
        orig_shape = coords.shape
        B = orig_shape[0]

        # 如果是4维，把中间两个维度合并
        if len(orig_shape) == 4:
            B, N1, N2, D = orig_shape
            coords = coords.view(B, N1 * N2, D)  # [B, N1*N2, D]
        elif len(orig_shape) == 3:
            B, N, D = orig_shape
        else:
            raise ValueError("coords shape must be [B,N,D] or [B,N,N,D]")

        coords = coords.clone()

        if self.normalize and input_range is not None:
            coords = shift_scale_points(coords, src_range=input_range)

        ndim = num_channels // coords.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        rems = num_channels - (ndim * coords.shape[2])

        final_embeds = []
        prev_dim = 0
        for d in range(coords.shape[2]):
            cdim = ndim
            if rems > 0:
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=coords.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            raw_pos = coords[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2)  # [B, N or N1*N2, num_channels]

        # 如果是原来4维的输入，恢复形状
        if len(orig_shape) == 4:
            final_embeds = final_embeds.view(B, N1, N2, -1)

        return final_embeds

    def forward(self, p1, p2):
        """
        计算每个点与其他点之间的距离和方向编码
        :param p: [B, N, 3]
        :return: [B, N, N, num_channels]
        """
        dist = torch.sqrt(pairwise_distance(p1, p2)).unsqueeze(-1)  # (B, N, N, 1)

        diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # [B, N, N, 3]

        if self.dist_mode == "angular":
            direction = diff / (torch.norm(diff, dim=-1, keepdim=True) + 1e-8)
        else:
            direction = diff / (dist + 1e-8)  # [B, N, N, 3]

        # 计算距离范围
        min_dist = dist.min().item()
        max_dist = dist.max().item()
        input_range_dist = (min_dist, max_dist)

        # 方向范围固定
        input_range_dir = (-1.0, 1.0)

        # 距离编码
        dist_enc = self.get_sine_embeddings(dist, num_channels=self.num_channels // 4, input_range=input_range_dist)
        # 方向编码 (x, y, z)
        dir_enc = self.get_sine_embeddings(direction, num_channels=self.num_channels // 4 * 3, input_range=input_range_dir)

        # [B, N, N, num_channels]
        return torch.cat([dist_enc, dir_enc], dim=-1)


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )







class DistanceDirectionMLPEncoding(nn.Module):
    def __init__(self, num_channels=32, dist_mode="euclidean", sep=False):

        super().__init__()
        self.num_channels = num_channels
        self.dist_mode = dist_mode
        self.sep = sep
        # distance branch (1 -> hidden -> out)

        if not sep:
            self.dist_mlp = nn.Sequential(
                nn.Linear(1, num_channels//2),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels//2, num_channels),
            )

            # direction branch (3 -> hidden -> out)
            self.dir_mlp = nn.Sequential(
                nn.Linear(3, num_channels//2),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels//2, num_channels)
            )

            self.fuse_fc = nn.Linear(num_channels * 2, num_channels)
            
        else:
            self.pos_mlp = nn.Sequential(
                nn.Linear(1+3, num_channels//2),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels//2, num_channels),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels, num_channels),
            )
    
    def compute_distance(self, p1, p2):
        diff = p2 - p1
        if self.dist_mode == "euclidean":
            return torch.norm(diff, dim=-1, keepdim=True)
        elif self.dist_mode == "sq_euclidean":
            return torch.sum(diff ** 2, dim=-1, keepdim=True)
        elif self.dist_mode == "angular":
            p1_norm = p1 / (torch.norm(p1, dim=-1, keepdim=True) + 1e-8)
            p2_norm = p2 / (torch.norm(p2, dim=-1, keepdim=True) + 1e-8)
            cos_theta = torch.clamp(torch.sum(p1_norm * p2_norm, dim=-1, keepdim=True), -1.0, 1.0)
            return torch.acos(cos_theta)
        else:
            raise ValueError(f"Unknown distance mode: {self.dist_mode}")


    def forward(self, p1, p2, scale='norm'):
        """
        计算每个点与其他点之间的距离和方向编码
        :param p: [B, N, 3]
        :return: [B, N, N, num_channels]
        """
        dist = torch.sqrt(pairwise_distance(p1, p2)).unsqueeze(-1)  # (B, N, N, 1)

        diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # [B, N, N, 3]

        if self.dist_mode == "angular":
            direction = diff / (torch.norm(diff, dim=-1, keepdim=True) + 1e-8)
        else:
            direction = diff / (dist + 1e-8)  # [B, N, N, 3]

        # if scale == 'log':
        #     dist = torch.log1p(dist)  # log(1 + dist)
        # elif scale == 'sqrt':
        #     dist = torch.sqrt(dist + 1e-6)
        # elif scale == 'norm':
        #     max_dist = dist.max()
        #     dist = dist / max_dist

        if not self.sep:
            dist = self.dist_mlp(dist)     
            direction = self.dir_mlp(direction)
            pos_embed = self.fuse_fc(torch.cat([dist, direction], dim=-1)) # [B, N, N, num_channels]
        else:
            pos_embed = torch.cat([dist, direction], dim=-1)
            pos_embed = self.pos_mlp(pos_embed)
        
        return pos_embed









class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )

class PositionRelationEmbedding(nn.Module):
    def __init__(self, num_heads=8, num_channels=64, temperature=10000, normalize=False, \
                 scale=None, dist_mode="euclidean", inplace=True, activation_layer=nn.ReLU, pos_emb_type='mlp', sep=False):
        """
        :param num_channels: 编码总通道数
        :param temperature: 正余弦频率缩放
        :param normalize: 是否归一化输入
        :param scale: 缩放比例
        :param dist_mode: 距离计算方式 ("euclidean" | "sq_euclidean" | "angular")
        """
        super().__init__()
        if pos_emb_type == 'sine':
            self.pos_emb = DistanceDirectionSineEncoding(num_channels, temperature, normalize, scale, dist_mode)
        elif pos_emb_type == 'mlp':
            self.pos_emb = DistanceDirectionMLPEncoding(num_channels, sep=sep)
        else:
            print('pos_emb_type error ...')
            exit()

        self.pos_proj = nn.Sequential(Conv2dNormActivation(
            num_channels,
            num_channels,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        ), Conv2dNormActivation(
            num_channels,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        ))

    def forward(self, p1, p2):
        pos_embed = self.pos_emb(p1, p2).permute(0,-1,1,2)
        pos_embed = self.pos_proj(pos_embed)
        return pos_embed

# ======================================距离+方向位置编码 END======================================






class DistanceDirectionTable(nn.Module):
    def __init__(self, num_channels=8, log_scale=2.0, rpe_quant='bilinear_5.2_10'):

        super().__init__()
        self.log_scale = log_scale

        self.interp_method, max_value, num_points = rpe_quant.split('_')
        num_points = int(num_points)
        relative_coords_table = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, num_points, dtype=torch.float32),  # dx
            torch.linspace(-1, 1, num_points, dtype=torch.float32),  # dy
            torch.linspace(-1, 1, num_points, dtype=torch.float32),  # dz
        ), dim=-1).unsqueeze(0)  # [1, num_dir, num_dir, num_dir, num_dist, 4]
        self.register_buffer("relative_coords_table", relative_coords_table)

        self.max_value = float(max_value)
        self.cpb_mlps = self.build_cpb_mlp(3, 128, num_channels)

        
    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True),
                                nn.ReLU(inplace=False),
                                nn.Linear(hidden_dim, out_dim, bias=False))
        return cpb_mlp
    
    def forward(self, p1, p2):
        """
        计算每个点与其他点之间的距离和方向编码
        :param p: [B, N, 3]
        :return: [B, num_channels, N, N]
        """
        B, nQ = p1.shape[:2]
        nK = p2.size(1)

        diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # [B, N1, N2, 3]

        deltas = torch.sign(diff) * torch.log2(torch.abs(diff)*self.log_scale + 1.0)
        deltas = deltas / self.max_value # b,n1,n2,3

        rpe_table = self.cpb_mlps(self.relative_coords_table).permute(0, 4, 1, 2, 3) # B, nH, 10, 10, 10

        rpe = F.grid_sample(rpe_table, deltas.view(1, 1, 1, -1, 3).to(rpe_table.dtype), mode=self.interp_method) \
                        .squeeze().view(-1, B, nQ, nK).permute(1, 0, 2, 3)

        return rpe


