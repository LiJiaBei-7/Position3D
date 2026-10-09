
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings
    

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



class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d=0.2, sigma_a=15, angle_k=3, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings



def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds
    
    def get_point_cloud_input_range(self, xyz):
        """
        xyz: (B, N, 3) 点云坐标
        return: input_range: (B, 3, 2)，每个 batch 的 x/y/z 的 [min, max]
        """
        # min/max over dim=1 (points)
        min_vals = xyz.min(dim=1).values  # (B, 3)
        max_vals = xyz.max(dim=1).values  # (B, 3)

        # 拼接成 (B, 3, 2)
        # input_range = torch.stack([min_vals, max_vals], dim=2)  # (B, 3, 2)

        return [min_vals, max_vals]

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                return self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                if input_range is None:
                    input_range = self.get_point_cloud_input_range(xyz)
                return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += (
                f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
            )
        return st
    


class DualKeyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k1_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(embed_dim)
        self.output_dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x):
        # x: (B, L, C)
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)  # (B, L, H, D)
        x = x.transpose(1, 2)                            # (B, H, L, D)
        return x

    def forward(
        self, query, key1, value, key2=None,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=True
    ):
        """
        query, key1, key2, value: (B, L or S, C)
        attn_mask: (L, S) or (B * H, L, S)
        key_padding_mask: (B, S)
        return: attn_output (B, L, C), attn_weights, attn_logits
        """
        B, L, C = query.shape
        S = key1.shape[1]

        Q = self._reshape_heads(self.q_proj(query))     # (B, H, L, D)
        K1 = self._reshape_heads(self.k1_proj(key1))    # (B, H, S, D)
        if key2 is not None:
            K2 = self._reshape_heads(self.k2_proj(key2))    # (B, H, S, D)
        V  = self._reshape_heads(self.v_proj(value))    # (B, H, S, D)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Q, K1.transpose(-2, -1)) * scale  # (B, H, L, S)
        if key2 is not None:
            attn_logits2 = torch.matmul(Q, K2.transpose(-2, -1)) * scale
            attn_logits = attn_logits + attn_logits2

        _attn_logits = attn_logits

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (L, S) → (1, 1, L, S)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                if attn_mask.shape[0] == B and attn_mask.shape[1] == L and attn_mask.shape[2] == S:
                    # (B, L, S) → (B, 1, L, S)
                    attn_mask = attn_mask.unsqueeze(1)  # broadcast across heads
                elif attn_mask.shape[0] == B * self.num_heads:
                    attn_mask = attn_mask.view(B, self.num_heads, L, S)
                else:
                    raise ValueError(f"Unsupported attn_mask shape: {attn_mask.shape}")
            else:
                raise ValueError(f"attn_mask must be (L, S), (B, L, S), or (B*num_heads, L, S)")

            attn_logits = attn_logits.masked_fill(attn_mask, float('-inf'))

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, H, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, C)
        attn_output = self.out_proj(attn_output)  # (B, L, C)

        attn_output = self.output_dropout(attn_output)
        attn_output = attn_output + query
        attn_output = self.norm(attn_output)

        if need_weights:
            return attn_output, attn_weights.mean(1), _attn_logits.mean(1)
        else:
            return attn_output


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.qpos_proj = nn.Linear(embed_dim, embed_dim)
        self.kpos_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

        # 初始化权重
        self._reset_parameters()  # <<<<<< 这里调用


    def _reset_parameters(self):
        # 用 Xavier 均匀初始化 Q/K/V 权重
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # 如果有 bias，就初始化为 0
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def _reshape_heads(self, x):
        # x: (B, L, C)
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)  # (B, L, H, D)
        x = x.transpose(1, 2)                            # (B, H, L, D)
        return x

    def forward(
        self, query, key, value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=True,
        pos_mask=None,
        pos_q=None,
        pos_k=None,
    ):
        """
        query, key1, key2, value: (B, L or S, C)
        attn_mask: (L, S) or (B * H, L, S)
        key_padding_mask: (B, S)
        pos_mask: (B,H,L,S)
        return: attn_output (B, L, C), attn_weights, attn_logits
        """
        B, L, C = query.shape
        S = key.shape[1]

        if pos_q is not None and pos_k is not None:
            query = query + self.qpos_proj(pos_q)
            key = key + self.kpos_proj(pos_k)

        Q = self._reshape_heads(self.q_proj(query))     # (B, H, L, D)
        K = self._reshape_heads(self.k_proj(key))    # (B, H, S, D)
        V  = self._reshape_heads(self.v_proj(value))    # (B, H, S, D)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, L, S)

        if pos_mask is not None:
            attn_logits += pos_mask

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (L, S) → (1, 1, L, S)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                if attn_mask.shape[0] == B and attn_mask.shape[1] == L and attn_mask.shape[2] == S:
                    # (B, L, S) → (B, 1, L, S)
                    attn_mask = attn_mask.unsqueeze(1)  # broadcast across heads
                elif attn_mask.shape[0] == B * self.num_heads:
                    attn_mask = attn_mask.view(B, self.num_heads, L, S)
                else:
                    raise ValueError(f"Unsupported attn_mask shape: {attn_mask.shape}")
            else:
                raise ValueError(f"attn_mask must be (L, S), (B, L, S), or (B*num_heads, L, S)")

            attn_logits = attn_logits.masked_fill(attn_mask, float('-inf'))

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        src_attn_logits = attn_logits

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, H, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, C)
        attn_output = self.out_proj(attn_output)  # (B, L, C)

        if need_weights:
            return attn_output, attn_weights.mean(1), src_attn_logits.mean(1)
        else:
            return attn_output




from .position_enc import DistanceDirectionTable
class _AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, log_scale=2.0, rpe_quant='bilinear_5.2_10'):  #log_scale=2.0, rpe_quant='bilinear_5.2_10'
        super().__init__()

        self.mha = MultiheadAttention(embed_dim, num_heads, dropout, bias)
        self.output_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.rpe_table = DistanceDirectionTable(num_heads, log_scale=log_scale, rpe_quant=rpe_quant)

    def forward(
        self, query, key, value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=True,
        pos_mask=None,
        pos_q=None,
        pos_k=None,
        p1=None,
        p2=None,
    ):

        if pos_mask is None:
            pos_mask = self.rpe_table(p1, p2)
            
        attn_output, attn_weight, src_attn_logits = self.mha(query, key, value, \
                                                                attn_mask=attn_mask, key_padding_mask=key_padding_mask, \
                                                                pos_mask=pos_mask, pos_k=pos_k, pos_q=pos_q, \
                                                                need_weights=need_weights)
        attn_output = self.output_dropout(attn_output)
        output = self.norm(query + attn_output)
        if need_weights:
            return output, attn_weight, src_attn_logits
        else:
            return output




from .position_enc import DistanceDirectionTable
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, log_scale=2.0, rpe_quant='bilinear_5.2_10'):  #log_scale=2.0, rpe_quant='bilinear_5.2_10'
        super().__init__()

        self.mha = MultiheadAttention(embed_dim, num_heads, dropout, bias)
        self.output_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.rpe_table = DistanceDirectionTable(num_heads, log_scale=log_scale, rpe_quant=rpe_quant)
        self.lang_modulated = LanguageModulated_1(num_heads, text_cond_dim=embed_dim)

        self.g_mlp = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 8), nn.Sigmoid()
        )

    def forward(
        self, query, key, value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=True,
        pos_mask=None,
        pos_q=None,
        pos_k=None,
        p1=None,
        p2=None,
        modulated_feats=None,
    ):

        if pos_mask is None:
            pos_mask = self.rpe_table(p1, p2) # (B, H, L, S)
        
        if modulated_feats is not None:
            if len(modulated_feats.shape) == 4:
                # # (B,N,N,d)-->(B,N,N,H)
                pos_mask = pos_mask * self.g_mlp(modulated_feats).permute(0,3,1,2)
            else:
                pos_mask = self.lang_modulated(pos_mask.permute(0,-2,-1,1), modulated_feats).permute(0,-1,1,2)
            
        attn_output, attn_weight, src_attn_logits = self.mha(query, key, value, \
                                                                attn_mask=attn_mask, key_padding_mask=key_padding_mask, \
                                                                pos_mask=pos_mask, pos_k=pos_k, pos_q=pos_q, \
                                                                need_weights=need_weights)
        attn_output = self.output_dropout(attn_output)
        output = self.norm(query + attn_output)
        if need_weights:
            return output, attn_weight, src_attn_logits
        else:
            return output


class LanguageModulated(nn.Module):
    def __init__(self, d_model, text_cond_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # γ^learn 和 β^learn 是可学习参数，形状取决于视觉特征维度
        self.gamma_learn = nn.Parameter(torch.ones(d_model))
        self.beta_learn = nn.Parameter(torch.zeros(d_model))
        # φ 是一个前馈网络，输入为条件文本特征 f_txt^cond
        self.phi = nn.Sequential(
            nn.Linear(text_cond_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, f_txt_cond):
        """
        x: [B, N, N, D]，B为batch，N为token数，D为维度
        f_txt_cond: [B, text_cond_dim]，条件文本特征

        return: F_modulated: [B, N, N, D]，调制后的特征
        """
        # φ(f_txt^cond): [B, D]
        phi_out = self.phi(f_txt_cond)

        # 计算 γ 和 β：[B, D]
        gamma = phi_out + self.gamma_learn
        beta = phi_out + self.beta_learn
        
        F_modulated = gamma.unsqueeze(1).unsqueeze(1) * x + beta.unsqueeze(1).unsqueeze(1)  # [B, N, N, D]

        return F_modulated


class LanguageModulated_1(nn.Module):
    """
    用文本特征对视觉特征进行调制 (FiLM 风格)
    vis_feat: [bs, n, n, d]
    text_feat: [bs, d]
    """
    def __init__(self, feature_dim, text_cond_dim):
        super().__init__()

        # 生成 gamma 和 beta 参数
        self.gamma_fc = nn.Linear(text_cond_dim, feature_dim)
        self.beta_fc = nn.Linear(text_cond_dim, feature_dim)

    def forward(self, vis_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        vis_feat: [bs, n, n, d]
        text_feat: [bs, d]
        """

        # 生成调制参数
        gamma = self.gamma_fc(text_feat).unsqueeze(1).unsqueeze(1)  # [bs,1,1,d]
        beta = self.beta_fc(text_feat).unsqueeze(1).unsqueeze(1)    # [bs,1,1,d]

        # FiLM 调制
        modulated_feat = vis_feat * gamma + beta
        return modulated_feat