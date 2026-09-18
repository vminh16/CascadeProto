"""
VIP-Seg: Visual Introspective Prototype Segmentation Network
for Point Cloud Few-Shot Semantic Segmentation

Paper: "Reasoning Beyond Points: A Visual Introspective Approach for Few-Shot 3D Segmentation"
NeurIPS 2025

Main Components:
1. Encoder-Decoder Backbone (Section 3.2)
2. DyPowerConv-Mamba Blocks for feature extraction
3. VIP Module for prototype refinement (Section 3.4)
4. Gating Fusion Network for multi-step prediction fusion (Section 3.4.4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder

class VIPSeg(nn.Module):
    def __init__(self, args):
        super(VIPSeg, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_points = args.pc_npts
        
        # ============================================================
        # Encoder-Decoder Backbone (Section 3.2)
        # ============================================================
        # The encoder consists of three stacked DyPowerConv-Mamba blocks
        # that integrate DyPowerConv for local geometric feature extraction
        # and Mamba blocks for capturing long-range dependencies
        
        self.encoder = Encoder(
            input_points=2048,
            num_stages=3,  # Three stages as described in paper
            embed_dim=60,   # Base embedding dimension
            k_neighbors=16, # K-NN neighbors for local grouping
            de_neighbors=10, # Neighbors for decoder upsampling
            alpha=1000,     # Wavelength hyperparameter for NTF
            beta=30,         # Amplitude hyperparameter for NTF
            num_experts=3    # HiConv
        )
        
        self.Dim = 900
        self.bn = nn.Sequential(nn.BatchNorm1d(self.Dim),
                                nn.ReLU())
        self.fc = nn.Sequential(nn.Conv1d(self.Dim, 196, 1),
                                nn.BatchNorm1d(196),
                                nn.ReLU(), 
                                nn.Conv1d(196, 128, 1), 
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                )
        

        self.num_steps = getattr(args, 'num_reasoning_steps', 4)  # Default: 4 steps (Table 3, Fig 3a)
        print(f" - Reasoning steps: {self.num_steps}")

        self.vip_module = nn.ModuleList()
        for step in range(self.num_steps):
            if step % 2 == 0:
                # Even steps: Prototype Enhancement Module
                self.vip_module.append(PrototypeEnhancementModule())
            else:
                # Odd steps: Prototype Difference Module
                self.vip_module.append(PrototypeDifferenceModule())
        
        # ============================================================
        # Gating Fusion Network (Section 3.4.4)
        # ============================================================
        # Learns importance weights for predictions from each reasoning step

        self.gating_network = GatingNetwork(input_dim=128, num_steps=self.num_steps)  # 门控网络

    def forward(self, support_x, support_y, query_x, query_y):
        N_way, K_shot, C, PN = support_x.shape

        support_x = support_x.float().cuda().permute(0, 1, 3, 2).view(-1, PN, C)
        support_y = support_y.float().cuda().view(N_way, K_shot, PN)
        query_x = query_x.float().cuda().permute(0, 2, 1).view(-1, PN, C)
        query_y = query_y.cuda().view(-1, PN)
        
        # Pass through the Parametric Encoder + Non-Parametric Decoder
        support_features = self.encoder(support_x)
        support_features = support_features / support_features.norm(dim=1, keepdim=True)
        
        query_features = self.encoder(query_x)
        query_features = query_features / query_features.norm(dim=1, keepdim=True)
        
        support_feat = self.bn(support_features)
        support_feat = self.fc(support_feat)
        support_feat = support_feat.permute(0, 2, 1)

        query_feat = self.bn(query_features)
        query_feat = self.fc(query_feat)
        query_feat = query_feat.permute(0, 2, 1)
        
        # obtain prototype
        feature_memory_list, label_memory_list = [], []
        support_feat = support_feat.view(N_way, K_shot, PN, -1)
        
        # ============================================================
        # Background Prototype (Class 0)
        # ============================================================
        # Extract background points (mask == 0) from all support samples

        mask_bg = (support_y == 0)
        bg_features = support_feat[mask_bg]
        
        if bg_features.shape[0] < 1:
            # If no background points, use small random prototype
            bg_features = torch.ones(1, support_feat.shape[-1]).cuda() * 0.1
        else:
            # Average all background points
            bg_features = bg_features.mean(0).unsqueeze(0)

        feature_memory_list.append(bg_features)
        label_memory_list.append(torch.tensor(0).unsqueeze(0))

        # ============================================================
        # Foreground Prototypes (Classes 1 to N)
        # ============================================================
        # Extract foreground points (mask == 1) for each class

        for i in range(N_way):
            mask_fg = (support_y[i] == 1)
            fg_features = support_feat[i, mask_fg]
            fg_features = fg_features.mean(0).unsqueeze(0)
            feature_memory_list.append(fg_features)
            label_memory_list.append(torch.tensor(i+1).unsqueeze(0))

        # Create one-hot labels for prototypes
        label_memory = torch.cat(label_memory_list, dim=0).cuda()
        label_memory = F.one_hot(label_memory, num_classes=N_way+1)

        # ============================================================
        # Concatenate and Normalize
        # ============================================================

        feature_memory = torch.cat(feature_memory_list, dim=0)
        feature_memory = feature_memory / torch.norm(feature_memory, dim=-1, keepdim=True)

        # Repeat for batch
        feature_memory = feature_memory.unsqueeze(0).repeat(N_way, 1, 1)

        # ============================================================
        # Multi-step Reasoning (Section 3.4.3)
        # ============================================================

        all_logits = []
        for step in range(self.num_steps):
            if step % 2 == 0:
                feature_memory = self.vip_module[step](query_feat, support_feat, feature_memory)
            else:
                feature_memory = self.vip_module[step](query_feat, support_feat, feature_memory) + feature_memory
            
            sim = [query_feat[i] @ feature_memory[i].t() for i in range(N_way)]
            sim = torch.stack(sim, dim=0)
            logits = sim @ label_memory.float()
            all_logits.append(logits)
        
        # Stack all logits
        all_logits = torch.stack(all_logits, dim=1)  # Shape: (N_way, num_steps, num_points, n_way+1)

        # ============================================================
        # Gating Fusion (Section 3.4.4, Eq. 21)
        # ============================================================
        # L_final = Σ(w_t · L_t)
        
        # Compute gating weights: [batch, num_steps]
        weights = self.gating_network(query_feat)  # Shape: (batch_size, num_steps)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, num_steps, 1, 1)
        
        # Weighted sum of predictions
        final_logits = (all_logits * weights).sum(dim=1)  # Shape: (N_way, num_points, n_way+1)
        
        # ============================================================
        # Compute Loss
        # ============================================================

        loss = F.cross_entropy(final_logits.reshape(-1, N_way+1), query_y.reshape(-1,).long())

        return final_logits, loss
    
    
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_steps):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_steps)
        
    def forward(self, query_feat):
        # query_feat: [batch_size, num_points, feature_dim]
        weights = F.softmax(self.fc(query_feat.mean(dim=1)), dim=-1)  # [batch_size, num_steps]
        return weights
    
class PrototypeEnhancementModule(nn.Module):
    """
    Prototype Enhancement Module (PEM)
    
    Enhances prototype discriminability through self-attention and cross-attention mechanisms.
    Reduces intra-class diversity by learning internal structural information and shared 
    information between support and query features.
    
    Reference: Section 3.4.1
    """
    
    def __init__(self):
        super(PrototypeEnhancementModule, self).__init__()
        self.in_channel = self.out_channel = 128

        # Max pooling for statistical characteristics extraction (Eq. 9)
        self.maxpool = nn.MaxPool1d(32, stride=32)

        self.layer_norm = nn.LayerNorm(self.in_channel)
        self.proj_dim = 72

         # Projection mapping W1 (Eq. 9)
        self.map = nn.Conv1d(64, self.proj_dim, 1, bias=False)

        # Prototype mapping
        self.proto_map = nn.Linear(self.in_channel, self.out_channel)

        # Transformation matrix W3 for self-correlation (Eq. 10)
        self.reweight = nn.Linear(self.in_channel, 1, bias=False)
        self.reweight_s = nn.Linear(self.in_channel, 1, bias=False)
        
        # Final transformation layers
        self.fc = nn.Linear(self.in_channel, self.out_channel, bias=False)
        self.fc_qs = nn.Linear(self.in_channel, self.out_channel, bias=False)

        # Layer normalization
        self.layer_norm_qs = nn.LayerNorm(self.in_channel) 

    def forward(self, query, supports, prototype):
        # print("?????", query.shape, supports.shape, prototype.shape)
        # torch.Size([2, 2048, 128]) torch.Size([2, 1, 2048, 128]) torch.Size([2, 3, 128])
        
        nway, kshot, PN, dim = supports.shape
        batch = query.shape[0]
        way = nway + 1 # including background
        residual = prototype
        
        supports = torch.cat([supports.mean(0).unsqueeze(0), supports], dim=0).reshape(-1, PN, dim) # [3, 2048, 128]
        
        # Extract statistical characteristics via max pooling (Eq. 9)
        # F'_q = MaxPool(Fq) · W1
        query = self.maxpool(query.transpose(1, 2)).transpose(1, 2) #[2, 64, 128]
        supports = self.maxpool(supports.transpose(1, 2)).transpose(1, 2) #[3, 64, 128]
        supports = supports.reshape(way, kshot, -1, dim)
        
        proto = 0
        for i in range(kshot): 
            support = supports[:, i, :, :] #[3, 64, 128]

             # Apply projection mapping (Eq. 9)
            que = self.map(query) #[2, 72, 128]
            sup = self.map(support) #[3, 72, 128]
            new_proto = self.proto_map(prototype) #[2, 3, 128]
            
            # ============================================================
            # Self-correlation Enhancement (Eq. 10-11)
            # ============================================================
            
            # Compute self-correlation matrices: A_s = W3(F'^T_s F'_s), A_q = W3(F'^T_q F'_q)
            que_G, sup_G = que.transpose(1,2) @ que, sup.transpose(1,2) @ sup ##[2, 128, 128] #[3, 128, 128]
            
            # Apply transformation W3 and scaling
            selfcor_q = self.reweight(que_G.unsqueeze(1)).squeeze(-1) / (128. ** 0.5) #[2, 3, 128]
            selfcor_s = self.reweight_s(sup_G.unsqueeze(0)).squeeze() / (128. ** 0.5) #[2, 3, 128]

            # Apply sigmoid activation and element-wise multiplication (Eq. 11)
            # F^self_p = Softmax(A_s)Fp + Softmax(A_q)Fp
            proto_self_q = torch.sigmoid(selfcor_q) * new_proto #[2, 3, 128]
            proto_self_s = torch.sigmoid(selfcor_s) * new_proto #[2, 3, 128]

            proto_self = self.fc_qs(proto_self_s) + self.fc_qs(proto_self_q) # [2, 3, 128]
            proto_self = self.layer_norm_qs(proto_self) # [2, 3, 128]
            
            # ============================================================
            # Cross-correlation Enhancement (Eq. 12-13)
            # ============================================================
            
            # Reshape for cross-correlation computation
            que, sup = que.reshape(self.proj_dim, -1), sup.reshape(self.proj_dim, -1) #[72, 256] [72, 384]

            # Compute cross-correlation: A_cross = F'^T_q F'_s (Eq. 12)
            crosscor = torch.matmul(que.transpose(0, 1) / (128. ** 0.5), sup) #[256, 384]

             # Reshape and apply softmax
            crosscor = crosscor.reshape(batch, dim, way, dim).permute(0, 2, 1, 3) # [2, 3, 128, 128]
            crosscor = F.softmax(crosscor, dim=-1) # [2, 3, 128, 128]

            # Apply cross-correlation to prototype (Eq. 13)
            # F^cross_p = Softmax(A_cross) ⊙ Fp
            proto_cross = torch.matmul(crosscor, new_proto.unsqueeze(2).transpose(-2, -1)).squeeze(-1) # [2, 3, 128]
            
            # ============================================================
            # Integration (Eq. 14)
            # ============================================================
            
            # Combine self and cross enhanced prototypes
            # F^e_p = F^self_p + F^cross_p + Fp
            output = self.fc(proto_cross + proto_self) # [2, 3, 128]
            output = self.layer_norm(output + residual) # [2, 3, 128]

            # Accumulate across k-shot samples
            proto = proto + output / kshot # [2, 3, 128]
            
        return proto
    
class PrototypeDifferenceModule(nn.Module):
    """
    Prototype Difference Module (PDM)
    
    Learns common representations from differences between support and query feature 
    distributions to eliminate domain gaps and semantic inconsistencies.
    
    Reference: Section 3.4.2
    """
    
    def __init__(self):
        super(PrototypeDifferenceModule, self).__init__()
        self.in_channel = self.out_channel = 128
        
        self.maxpool = nn.MaxPool1d(32, stride=32)

        self.layer_norm = nn.LayerNorm(self.in_channel)
        self.proj_dim = 72
        
        self.map = nn.Conv1d(64, self.proj_dim, 1, bias=False)
        self.proto_map = nn.Linear(self.in_channel, self.out_channel)
        
        self.reweight = nn.Linear(self.in_channel, 1, bias=False)

        self.fc = nn.Linear(self.in_channel, self.out_channel, bias=False)

    def forward(self, query, supports, prototype):
        # print("?????", query.shape, supports.shape, prototype.shape)
        # torch.Size([2, 2048, 128]) torch.Size([2, 1, 2048, 128]) torch.Size([2, 3, 128])
        
        nway, kshot, PN, dim = supports.shape
        batch = query.shape[0]
        way = nway + 1
        residual = prototype
        
        supports = torch.cat([supports.mean(0).unsqueeze(0), supports], dim=0).reshape(-1, PN, dim)

         # Apply max pooling
        query = self.maxpool(query.transpose(1, 2)).transpose(1, 2)
        supports = self.maxpool(supports.transpose(1, 2)).transpose(1, 2)

        # Reshape to separate classes and shots
        supports = supports.reshape(way, kshot, -1, dim)
        
        proto = 0
        for i in range(kshot): 
            support = supports[:, i, :, :]

            # Apply projection
            que = self.map(query)
            sup = self.map(support)
            new_proto = self.proto_map(prototype)
            
            # ============================================================
            # Difference Learning (Eq. 15-16)
            # ============================================================
            
            # Compute feature correlation matrices
            que_G, sup_G = que.transpose(1,2) @ que, sup.transpose(1,2) @ sup

            # Calculate difference information: ΔG = F'^T_q F'_q - F'^T_s F'_s (Eq. 15)
            delta_G = que_G.unsqueeze(1) - sup_G.unsqueeze(0)

            # Apply transformation and scaling
            selfcor = self.reweight(delta_G).squeeze() / (128. ** 0.5)

            # Adjust prototype with difference information (Eq. 16)
            # F^delta_p = sigmoid(ΔG) ⊙ F^e_p
            proto_self = torch.sigmoid(selfcor) * new_proto
            
            # ============================================================
            # Cross-attention Enhancement (Eq. 17)
            # ============================================================
            
            # Reshape for cross-correlation
            que, sup = que.reshape(self.proj_dim, -1), sup.reshape(self.proj_dim, -1)

            # Compute cross-correlation
            crosscor = torch.matmul(que.transpose(0, 1) / (128. ** 0.5), sup)
            crosscor = crosscor.reshape(batch, dim, way, dim).permute(0, 2, 1, 3)
            crosscor = F.softmax(crosscor, dim=-1)

            # F^e_cross_p = Softmax(A_cross) ⊙ F^e_p (Eq. 17)
            proto_cross = torch.matmul(crosscor, new_proto.unsqueeze(2).transpose(-2, -1)).squeeze(-1)

            # ============================================================
            # Final Integration (Eq. 18)
            # ============================================================
            output = self.fc(proto_cross + proto_self)
            output = self.layer_norm(output + residual)
            
            proto = proto + output / kshot
            
        return proto