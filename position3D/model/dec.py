import torch
import torch.nn as nn
from .sample_model import SamplingModule
import torch.nn.functional as F
from ..torch.nn import MultiheadAttention

from pointnet2.pointnet2_utils import GatherOperation
from .module import AttentionLayer, LanguageModulated, MLP, PositionEmbeddingCoordsSine

import h5py


def get_input_range(points):
    min_xyz, _ = points.min(dim=1)  # (bsz, 3)
    max_xyz, _ = points.max(dim=1)  # (bsz, 3)
    return [min_xyz, max_xyz]



class ThreeLayerMLP(nn.Module):
    """A 3-layer MLP with normalization and dropout."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x):
        """Forward pass, x can be (B, dim, N)."""
        return self.net(x)

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_mask=None, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        B = query.shape[0]
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, query.shape[1], k.shape[1])
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=batch_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=batch_mask)
        self.dropout(output)
        output = output + query
        self.norm(output)

        return output, output_weight, src_weight # (b, n_q, d_model), (b, n_q, n_v)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0, glu = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = glu
        if glu:
            self.glu_p = nn.Linear(d_model, d_model)
            self.glu_g = nn.Linear(d_model, d_model)
        
        self.qpos_proj = nn.Linear(d_model, d_model)
        self.kpos_proj = nn.Linear(d_model, d_model)

    def with_pos_embed(self, q, k, pos):
        if pos is not None:
            q = self.qpos_proj(pos) + q
            k = self.kpos_proj(pos) + k
            return q, k
        else:
            return q, k 

    def forward(self, x, x_mask=None, attn_mask=None, pe=None, norm=True):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q,k = self.with_pos_embed(x, x, pe)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, q.shape[1], k.shape[1])
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        if self.glu:
            output = self.glu_p(output) * (self.glu_g(output).sigmoid())
        output = self.dropout(output) + x
        if norm: output = self.norm(output)
        return output

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, norm=True):
        output = self.net(x)
        output = output + x
        if norm: output = self.norm(output)
        return output
    
class DEC(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """
    def __init__(
        self,
        num_layer=6,
        num_class=256,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
        sampling_module=None,
        kernel='top1',
        global_feat='mean',
        lang_att=False,
        contrastive_align_loss=False,
        pos_channel=32,
        lang_modulated=False,
        seed_layer=3,
        query_pos_type='mask',
        use_pos=False,
        log_scale=2.0, 
        rpe_quant='bilinear_5.2_10',
        local_K=[32,16,8,8]
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_class = num_class
        self.d_model = d_model
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        
        self.input_proj_2d = nn.Sequential(nn.Linear(1024, d_model*2),nn.ReLU(),nn.Linear(d_model*2, d_model),nn.ReLU(),nn.Linear(d_model, d_model))#clip
        self.sum_norm = nn.LayerNorm(d_model)
        
        self.lang_att = lang_att
        self.contrastive_align_loss = contrastive_align_loss

        H = 768
        self.lang_proj = nn.Linear(H, d_model)
        self.lang_norm = nn.LayerNorm(d_model)
        
        if sampling_module is not None:
            self.sampling_module = SamplingModule(**sampling_module)
        else:
            self.sampling_module = None

        self.query_generator = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model))
        
        # DDI and SWA
        self.swa_layers = nn.ModuleList([])
        self.rra_layers = nn.ModuleList([])
        self.rla_layers = nn.ModuleList([])
        self.swa_ffn_layers = nn.ModuleList([])
        self.sem_cls_heads = nn.ModuleList([])
        self.scg = nn.ModuleList([])
        self.lqg = nn.ModuleList([])
        self.query_ffn_layers = nn.ModuleList([])
        self.local_norm = nn.ModuleList([])

        # local  attention
        self.local_attention_feat = nn.ModuleList([])
        self.local_attention_pos = nn.ModuleList([])
        self.neighbor_mlp = nn.ModuleList([])

        if self.lang_att:
            self.lla_layers = nn.ModuleList([])
            self.lsa_layers = nn.ModuleList([])
            self.lsa_ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.swa_layers.append(AttentionLayer(embed_dim=d_model, num_heads=nhead, dropout=dropout, log_scale=log_scale, rpe_quant=rpe_quant))
            self.rra_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.rla_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.swa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.sem_cls_heads.append(ThreeLayerMLP(d_model, self.num_class))
            self.scg.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.query_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.local_norm.append(nn.LayerNorm(d_model))

            self.local_attention_feat.append(MLP(d_model, d_model//2, 1, 2))
            self.local_attention_pos.append(MLP(3, d_model//2, 1, 2))
            self.neighbor_mlp.append(MLP(d_model, d_model, d_model, 2))

            # self.lqg.append(CrossAttentionLayer(d_model, nhead, dropout))
            if self.lang_att:
                self.lla_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
                self.lsa_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
                self.lsa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))    
        
        self.sem_cls_head = ThreeLayerMLP(d_model, self.num_class)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        
        
        self.indi_embedding = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2), nn.Linear(2, 2))
        self.indi_norm = nn.LayerNorm(d_model)

        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.kernel = kernel
        self.global_feat = global_feat
        
        self.k = 8

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_vision = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, 64))
            self.contrastive_align_projection_text = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, 64))         
        
        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)

        self.seed_attention = nn.ModuleList([])
        self.seed_ffn = nn.ModuleList([])
        self.seed_layer = seed_layer
        for i in range(self.seed_layer):
            self.seed_attention.append(AttentionLayer(embed_dim=d_model, num_heads=nhead, dropout=dropout, log_scale=log_scale, rpe_quant=rpe_quant))
            self.seed_ffn.append(FFN(d_model, hidden_dim, dropout, activation_fn)) 

        self.LanguageModulated = lang_modulated
        if LanguageModulated:
            self.lang_modulated = LanguageModulated(pos_channel, d_model)  


        self.out_bbox = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 3))
        nn.init.constant_(self.out_bbox[-1].weight.data, 0)
        nn.init.constant_(self.out_bbox[-1].bias.data, 0)
        self.bbox_embed = MLP(d_model, d_model, 3, 3)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.query_pos_type = query_pos_type
        self.position_embedding = PositionEmbeddingCoordsSine(d_pos=d_model)
        self.key_position_embedding = PositionEmbeddingCoordsSine(d_pos=d_model, normalize=True)
        self.ref_point_head = MLP(d_model, d_model, d_model, 2)
        self.use_pos = use_pos

        self.local_K = local_K


    def calculate_fine_grained_ambiguity(self, word_feats, query_feats, temperature=0.01):
        """
        计算细粒度(单词级别)对齐下的文本模糊度
        
        参数:
        word_feats: shape [B, N, D], N 是单词数量
        query_feats: shape [B, Q, D], Q 是候选目标 (Queries) 数量
        temperature: 温度系数，用于放大余弦相似度的差异。推荐范围 0.01 ~ 0.1
        """
        # 1. 特征 L2 归一化 (计算余弦相似度的标准操作)
        word_feats = F.normalize(word_feats, p=2, dim=-1)
        query_feats = F.normalize(query_feats, p=2, dim=-1)
        
        # 2. 计算 单词-Query 相似度矩阵
        # [B, N, D] 乘以 [B, D, Q] -> 结果形状 [B, N, Q]
        # sim_matrix[b, n, q] 代表第 n 个单词和第 q 个目标的相似度
        sim_matrix = torch.bmm(word_feats, query_feats.transpose(1, 2))
        
        # 3. 聚合得到 Query 级别的最终得分 (在单词维度 N 上取最大值)
        # 物理意义：只要文本中有任何一个词强烈指向该 Query，该 Query 的得分就高
        # 结果形状: [B, Q]
        query_scores = torch.mean(sim_matrix, dim=1)
        
        # 4. 引入温度系数并计算概率分布
        # 因为余弦相似度在 [-1, 1] 之间，差异很小，必须用小温度系数放大
        probs = F.softmax(query_scores / temperature, dim=-1)
        
        # 5. 在 Q 个目标候选上计算信息熵
        # 结果形状: [B]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # 6. 计算平滑的 Loss 权重，并在 Batch 内部进行相对归一化
        weight = 1.0 / (1.0 + entropy)
        # weight = weight / (weight.mean() + 1e-8)
        
        return entropy, weight, probs

   
    def forward_iter_pred(self, x, fps_seed_sp, batch_offsets, lang_feats=None, lang_masks=None, \
                          sp_pos=None, feats_2d=None, scan_ids=None, ann_ids=None, knn_idx=None):
        """
        x [B*M, inchannel]
        """
        
        lang_feats = self.lang_proj(lang_feats) 
        lang_feats = self.lang_norm(lang_feats)  
        lang_masks = ~(lang_masks.bool())
        inst_feats = self.input_proj(x)
        
        inst_feats_2d = self.input_proj_2d(feats_2d)
        inst_feats = self.sum_norm(inst_feats+inst_feats_2d)
        
        mask_feats = self.x_mask(x)
        inst_feats, batch_mask = self.get_batches(inst_feats, batch_offsets)
        mask_feats, _ = self.get_batches(mask_feats, batch_offsets)
        
        sp_pos, _ = self.get_batches(sp_pos, batch_offsets)
        
        prediction_masks, prediction_scores, prediction_classes, prediction_indis, prediction_box = [], [], [], [], []
        
        sample_inds = None
        ref_scores = None
         
        seed_sp = inst_feats.gather(dim=1, index=fps_seed_sp.long().unsqueeze(-1).repeat(1, 1, inst_feats.size(-1)))
        seed_sp_pos = sp_pos.gather(dim=1, index=fps_seed_sp.long().unsqueeze(-1).repeat(1, 1, sp_pos.size(-1)))

        input_range = get_input_range(sp_pos)
        input_ranges_mins, input_ranges_maxs = input_range[0].unsqueeze(1), input_range[1].unsqueeze(1)


        # # CUDA gather: [B, D, N*K]
        B, N, K = knn_idx.shape
        D = inst_feats.size(-1)
        neighbor_feats = GatherOperation.apply(inst_feats.transpose(1, 2).contiguous(), knn_idx.int().reshape(knn_idx.size(0), -1).contiguous())
        # to [B, N, K, D]
        neighbor_feats = neighbor_feats.view(B, D, N, K).permute(0, 2, 3, 1)
        
        if self.use_pos:
            seed_sp_pos_norm = (seed_sp_pos - input_ranges_mins) / (input_ranges_maxs - input_ranges_mins)
            seed_pos = self.position_embedding(seed_sp_pos_norm)
        else:
            seed_pos = None

        # ==============seed update==============
        for i in range(self.seed_layer):
            seed_sp_g,_,_ = self.seed_attention[i](seed_sp,seed_sp,seed_sp, pos_mask=None, p1=seed_sp_pos, p2=seed_sp_pos, \
                                                   pos_q=seed_pos, pos_k=seed_pos)
            seed_sp_l = self.neighbor_aggregation(seed_sp, neighbor_feats)
            _seed_sp = seed_sp_g + seed_sp_l
            seed_sp = self.seed_ffn[i](_seed_sp) + seed_sp
        # ==============seed update==============

        scg_mask = self.get_scg_mask(seed_sp_pos)
        seed_sp = self.scg_head(seed_sp, attn_mask = ~scg_mask)

         # sampling
        if hasattr(self, 'sampling_module') and self.sampling_module is not None:
            sample_inds, ref_scores = self.sampling_module(seed_sp, lang_feats, None, lang_masks)
            sample_inds = sample_inds.long()
            sampled_seed = seed_sp.gather(dim=1, index=sample_inds.unsqueeze(-1).repeat(1, 1, seed_sp.size(-1)))
            query_pos = seed_sp_pos.gather(dim=1, index=sample_inds.unsqueeze(-1).repeat(1, 1, seed_sp_pos.size(-1)))
            query = self.query_generator(sampled_seed)
        else:
            query = self.query_generator(seed_sp)
            query_pos = seed_sp_pos

        
        # with h5py.File('visualization/data/G_sampling_points_coords.hdf5', 'a') as f:
        #         for a in range(len(ann_ids)):
        #             name = f'{scan_ids[0]}_{ann_ids[a]}'
        #             f.create_dataset(f"{name}_seed_pos", data=seed_sp_pos[a].detach().cpu())
        #             f.create_dataset(f"{name}_query_pos", data=query_pos[a].detach().cpu())
        #             f.create_dataset(f"{name}_sample_inds", data=sample_inds[a].detach().cpu())


            
        scg_mask = self.get_scg_mask(query_pos)

        proj_queries = []
        if self.contrastive_align_loss:
            proj_queries.append(F.normalize(self.contrastive_align_projection_vision(query), p=2, dim=-1))
        else:
            proj_queries.append(None)
            
        proj_tokens = []
        if self.contrastive_align_loss:
            proj_tokens.append(F.normalize(self.contrastive_align_projection_text(lang_feats), p=2, dim=-1))
        else:
            proj_tokens.append(None)
        
        
        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)

        pred_indis = self.indi_embedding(query)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)
        prediction_indis.append(pred_indis)
        prediction_box.append(torch.tensor(-1).cuda())

        # ========sp positional encoding========
        key_pos = self.key_position_embedding(sp_pos, input_range=input_range)
        reference_points = (query_pos - input_ranges_mins) / (input_ranges_maxs - input_ranges_mins)


        # ================================================================
        # ======================= decoding ===============================
        # ================================================================
                
        # multi-round
        l = 0
        lang_query = lang_feats

        local_K = self.local_K
        for i in range(self.num_layer):
            # lang_query, w1, w2 = self.lqg[i](query, lang_feats)
        
            if i>l:
                lang_query = lang_query + lang_feats
            
            if self.lang_att:
                lang_query = self.lla_layers[i](lang_query, lang_masks)
                lang_query, _, _ = self.lsa_layers[i](inst_feats, lang_query, batch_mask, None)
                lang_query = self.lsa_ffn_layers[i](lang_query)

            # reference_points是归一化后的坐标值
            obj_center = reference_points[..., :3]
            if self.use_pos:
                query_sine_embed = self.position_embedding(obj_center)
                query_pos = self.ref_point_head(query_sine_embed)
            else:
                query_pos = key_pos = None

            reference_points_coords_float = reference_points * (input_ranges_maxs - input_ranges_mins) + input_ranges_mins
            
            lang_mean = (lang_query * ~lang_masks.unsqueeze(-1)).sum(1) / (~lang_masks).sum(-1, keepdim=True) # [B, C]

            if local_K[i] == 0:
                select_mask = None
            else:
                select_mask = ~self.get_scg_mask(reference_points_coords_float, sp_pos, batch_mask, k=local_K[i])
            query_g, attn_weights, _ = self.swa_layers[i](query, inst_feats, inst_feats, \
                                                        key_padding_mask=batch_mask, attn_mask=select_mask, pos_mask=None, \
                                                        pos_q=query_pos, pos_k=key_pos, p1=reference_points_coords_float, p2=sp_pos, \
                                                        modulated_feats=lang_mean)

            query = self.query_ffn_layers[i](query_g + self.local_norm[i](query))
            
            query_rra = self.rra_layers[i](query)
            query_rla, _, _ = self.rla_layers[i](lang_query, query, lang_masks)
            
            if self.lang_att:
                lang_query = self.lla_layers[i](lang_query, lang_masks)
                lang_query, _, _ = self.lsa_layers[i](query, lang_query)
                lang_query = self.lsa_ffn_layers[i](lang_query)

            query = query + query_rla + query_rra 
            
            query = self.scg[i](query, attn_mask=~scg_mask)
            query = self.swa_ffn_layers[i](query)

            # ==========================reference_points的预测==========================
            if self.query_pos_type == 'box' and i != self.num_layer -1:
            # xyz_range[0] max ; xyz_range[1] min
                obj_center_offset = self.bbox_embed(query)
                unnorm_reference_points = obj_center * (input_ranges_maxs - input_ranges_mins) + input_ranges_mins + obj_center_offset
                # 归一化
                norm_reference_points = (unnorm_reference_points - input_ranges_mins) / (input_ranges_maxs - input_ranges_mins) #[num_queries, bsz, 3]
                reference_points = norm_reference_points.detach()

            pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)

            if i != self.num_layer -1:
                if self.query_pos_type == 'box':
                    pred_bboxes = self.out_bbox(query)
                    pred_bboxes = norm_reference_points * (input_ranges_maxs - input_ranges_mins) + input_ranges_mins  + pred_bboxes
                else:
                    pred_bboxes = norm_reference_points * (input_ranges_maxs - input_ranges_mins) + input_ranges_mins


                # with h5py.File(f'visualization/data/G_pred_centers_{i}.hdf5', 'a') as f:
                #     for a in range(len(ann_ids)):
                #         name = f'{scan_ids[0]}_{ann_ids[a]}'
                #         f.create_dataset(f'{name}', data=pred_bboxes[a].detach().cpu())


        
            pred_indis = self.indi_embedding(query)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
            prediction_indis.append(pred_indis)
            if i != self.num_layer -1:
                prediction_box.append(pred_bboxes)

            if self.contrastive_align_loss:
                query_sem = F.normalize(self.contrastive_align_projection_vision(query), p=2, dim=-1)
                proj_queries.append(query_sem)
            else:
                proj_queries.append(None)
                
            if self.contrastive_align_loss:
                lang_sem = F.normalize(self.contrastive_align_projection_text(lang_query), p=2, dim=-1)
                proj_tokens.append(lang_sem)
            else:
                proj_tokens.append(None)


            # if i == self.num_layer -1:
            #     entropy, weight, _ = self.calculate_fine_grained_ambiguity(self.contrastive_align_projection_text(lang_query), self.contrastive_align_projection_vision(query))
                

            #     with open('visualization/data/uncertainty.txt', 'a') as f:
            #         for a in range(len(ann_ids)):
            #             name = f'{scan_ids[0]}_{ann_ids[a]}'
            #             f.write(f'{name} {entropy[a].item()} {weight[a].item()} \n')

            # ------------------------------
            # score = query_sem @ (lang_sem * ~lang_masks.unsqueeze(-1)).transpose(-1,-2)
            # score = torch.softmax(score  / 0.07, 1).sum(-1) # [B, num_query]

            # if i == self.num_layer -1:
            #     for a in range(len(ann_ids)):
            #         name = f'{scan_ids[0]}_{ann_ids[a]}'
            #         if name == 'scene0011_00_16':
            #             indicator = F.softmax(pred_indis[a], dim=-1)[:,1]
            #             values, indices = torch.sort(indicator, descending=True)
            #             print(values)

            # with h5py.File(f'visualization/data/G_attn_weight_{i}.hdf5', 'a') as f:
            #     for a in range(len(ann_ids)):
            #         name = f'{scan_ids[0]}_{ann_ids[a]}'
            #         f.create_dataset(f'{name}_attn', data=attn_weights[a].detach().cpu())
            #         f.create_dataset(f'{name}_pred_indis', data=pred_indis[a].detach().cpu())
            #         f.create_dataset(f'{name}_pad_mask', data=batch_mask[a].detach().cpu())
            #         f.create_dataset(f'{name}_pred_score', data=pred_bboxes[a].detach().cpu())
            #         f.create_dataset(f'{name}_unnorm_reference_points', data=unnorm_reference_points[a].detach().cpu())
            #         f.create_dataset(f'{name}_pred_mask', data=pred_masks[a].detach().cpu())
            #         f.create_dataset(f'{name}_pred_bboxes', data=pred_bboxes[a].detach().cpu())

            
        prediction_box = prediction_box[:-1]
        prediction_box.append(torch.tensor(-1).cuda())

        # with h5py.File('visualization/data/sp_float.hdf5', 'a') as f:
        #     for a in range(len(ann_ids)):
        #         name = f'{scan_ids[0]}_{ann_ids[a]}'
        #         f.create_dataset(f"{name}", data=sp_pos[a].detach().cpu())

        # pred_indis = torch.stack(prediction_indis, dim=0).mean(dim=0) 
        # pred_masks = torch.stack(prediction_masks, dim=0).mean(dim=0) 

            # torch.cuda.empty_cache()
        return {
            'masks': pred_masks,
            'batch_mask': batch_mask,
            'scores': pred_scores,
            # 'scores_lang': score,
            'indis': pred_indis, # [B, B_q, 2]
            'proj_queries': proj_queries[-1],
            'proj_tokens': proj_tokens[-1],
            'sample_inds': sample_inds, # [B, K]
            'ref_scores': ref_scores, # [B, M]
            'pred_box': pred_bboxes,
            'aux_outputs': [{
                'masks': a,
                'scores': b,
                'proj_queries': c,
                'indis': d,
                'proj_tokens': e,
                'pred_box': f,
            } for a, b, c, d, e, f in zip(
                prediction_masks[:-1],
                prediction_scores[:-1],
                proj_queries[:-1],
                prediction_indis[:-1],
                proj_tokens[:-1], 
                prediction_box,
            )],
        }


  
    def neighbor_aggregation(self, x, neighbor):
        # x: b,n,d
        # neighobr: b,n,k,d
        x = x.unsqueeze(-2)
        score = torch.einsum('bnld,bnkd->bnlk', x, neighbor) # b,n,1,k
        score = torch.softmax(score, dim=-1)
        neighbor_agg = torch.einsum('bnlk,bnkd->bnld', score, neighbor) # b,n,1,d
       
        return neighbor_agg.squeeze(-2)
    
    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        if torch.is_tensor(max_len):
            max_len = max_len.item()
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)
            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def get_mask(self, query, mask_feats, batch_mask):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        if self.attn_mask:
            attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None
        return pred_masks, attn_masks

    def prediction_head(self, query, mask_feats, batch_mask):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)
        return pred_scores, pred_masks, attn_masks

    def get_scg_mask(self, pos_q, pos_k=None, k_mask=None, k=8):
        if pos_k == None: pos_k = pos_q
        scg_mask = torch.zeros((pos_q.shape[0],pos_q.shape[1],pos_k.shape[1]),device=pos_q.device).bool()
        dis = ((pos_q.unsqueeze(2)-pos_k.unsqueeze(1))**2).sum(-1)
        if k_mask is not None:
            dis = torch.masked_fill(dis, k_mask.unsqueeze(1).repeat(1,pos_q.shape[1],1), 1000000.0)
        ind = torch.topk(dis, k, dim=-1, largest=False)[1]

        # 向量化 scatter
        scg_mask.scatter_(2, ind, True)

        # for i in range(ind.shape[0]):
        #     for j in range(ind.shape[1]):
        #         scg_mask[i,j,ind[i][j]] = True
        return scg_mask

    def get_scg_mask_ratio(self, pos_q, pos_k=None, k_mask=None, ratio=0.8):
        if pos_k == None: pos_k = pos_q
        scg_mask = torch.zeros((pos_q.shape[0],pos_q.shape[1],pos_k.shape[1]),device=pos_q.device).bool()
        dis = ((pos_q.unsqueeze(2)-pos_k.unsqueeze(1))**2).sum(-1)
        if k_mask is not None:
            dis = torch.masked_fill(dis, k_mask.unsqueeze(1).expand(-1, pos_q.shape[1], -1), 1e6)
        len_k = k_mask.sum(-1)
        k = (len_k.float()/ratio).int()

        for i in range(scg_mask.shape[0]):
            for j in range(scg_mask.shape[1]):
                k_i = k[i]
                ind = torch.topk(dis[i][j], k_i, dim=-1, largest=False)[1]
                scg_mask[i,j,ind] = True

        return scg_mask
    
    def forward(self, x, fps_seed_sp, batch_offsets, lang_feats=None, lang_masks=None, sp_pos=None, feats_2d=None, scan_ids=None, ann_ids=None, knn_idx=None):
        if self.iter_pred:
            return self.forward_iter_pred(x, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_pos, feats_2d, scan_ids, ann_ids, knn_idx)
        else:
            raise NotImplementedError
