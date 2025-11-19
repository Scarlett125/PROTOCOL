import torch
from torch import nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mutual_info_score
from Cos_classifier import Cos_classifier

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)



class View_Weight(nn.Module):
    def __init__(self, num_views, Z_dims, U_dim):
        super(View_Weight, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(Z_dim, U_dim),
                nn.LeakyReLU(negative_slope=0.01)
            ) for Z_dim in Z_dims
        ])
        self.alpha = nn.Parameter(torch.full((num_views,), 1 / num_views), requires_grad=True)

    def forward(self, Z_list):
        U_num = 0
        alpha_normalized = F.softmax(self.alpha, dim=0)
        
        for v, Z_v in enumerate(Z_list):
            W_v_output = self.fc_layers[v](Z_v)  
            U_num += alpha_normalized[v] * W_v_output 
      
        U = U_num
        return U, alpha_normalized



class PROTOCOL(nn.Module):
    def __init__(self, view, num_heads,nclusters, input_size, low_feature_dim, high_feature_dim, device):
        super(PROTOCOL, self).__init__()
        self.encoders = []
        self.decoders = []
        nclusters = [nclusters]
        self.num_heads = num_heads
        if num_heads>1:
            nclusters = [nclusters[0]]*self.num_heads
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], low_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.view_dims = [low_feature_dim] * view
        self.consensus_dim = low_feature_dim
        self.weights = View_Weight(view, self.view_dims, self.consensus_dim)
        self.Specific_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(low_feature_dim, nclusters[0]),
            nn.Softmax(dim=1)
)
        self.label_contrastive_module_high = nn.Sequential(
            nn.Linear(high_feature_dim, nclusters[0]),
            nn.Softmax(dim=1)
)
        self.Common_view_C = nn.Sequential(
            nn.Linear(low_feature_dim*view, high_feature_dim),
        )
        self.Common_view_F = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )

        self.ot_predicted = nn.ModuleList([Cos_classifier(low_feature_dim, nclusters[i]) for i in range(self.num_heads)])
        self.view = view
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=low_feature_dim*view, nhead=1, dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
        
    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        qls = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.Specific_view(z), dim=1)
            ql = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
            qls.append(ql)
        return  xrs, zs, hs, qls     
    
    def forward_ot(self, xs):
        qs_ot = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q_ot= [ot_predicted(z) for ot_predicted in self.ot_predicted]
            qs_ot.append(q_ot)
        return qs_ot
                                                                                                                                                         
    def ViewFusion(self, xs):
        zs = []
        for v in range(self.view): 
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
        commonz_ori = torch.cat(zs, 1)
        commonz, S = self.TransformerEncoderLayer(commonz_ori)
        split_commonz = torch.split(commonz, split_size_or_sections=commonz.size(1) // self.view, dim=1)
        commonz_fused, weights_test = self.weights(split_commonz)  
        commonz_fused_high = normalize(self.Common_view_F(commonz_fused), dim=1)
        commonz_fused_high_qhs = self.label_contrastive_module_high(commonz_fused_high)

        
        return commonz_fused_high, commonz_fused_high_qhs, S,weights_test,split_commonz
    

    




