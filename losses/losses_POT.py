import torch
import torch.nn as nn
import torch.nn.functional as F
from mvc_ot.sinkhorn_knopp import SinkhornLabelAllocation, SemiCurrSinkhornKnopp
import itertools as it
from termcolor import colored
from losses.ramps import sigmoid_rampup
import numpy as np

EPS=1e-8

class POT_loss(nn.Module):
    def __init__(self,p,sk_type="sela", factor=10, num_iter=3, total_iter=100000,start_iter=0,logits_bank=None):
        super(POT_loss, self).__init__() 
        sk_iter_limit = p["sk_iter_limit"]
        self.num_heads=p["num_heads"]
        self.sk = [SemiCurrSinkhornKnopp(gamma=p["gamma_bound"], epsilon=factor, numItermax=sk_iter_limit) for _ in range(self.num_heads)]
        self.logits_bank=logits_bank
        self.sk_type=sk_type
        self.criterion=torch.nn.CrossEntropyLoss().cuda()
        self.labels=[[] for _ in range(self.num_heads)] 
        self.target=[] 
        self.i = start_iter
        self.total_iter = total_iter
        self.rho_base=p["rho_base"]
        self.rho_upper = p["rho_upper"] - p["rho_base"]
        self.rho_fix = p["rho_fix"]
        self.rho_strategy = p["rho_strategy"]
        self.label_quality_show = p["label_quality_show"]
        for sk in self.sk:
            sk.rho = p["rho_base"]
        self.prev_loss = None
        self.current_loss = None

    def forward(self, logits, target=None, data_idxs=None):
        batch_size=logits[0].shape[0]
        if not self.rho_fix:
            self.set_rho(self.i, self.total_iter)
        
        if self.logits_bank is None:
            if self.sk_type == "sla":
                assert data_idxs is not None, "data_idxs should not be None for SLA"
                pseudo_labels=[[self.sk[head_id](head, data_idxs) for head_id ,head in enumerate(view)] for view in logits]
            else:
                pseudo_labels=[self.sk[head_id](head) for head_id, head in enumerate(logits)]
        else:
            pseudo_labels=[]
            for view_id,view in enumerate(logits):
                pseudo_labels_view=[]
                for head_id,head in enumerate(view):
                    memory, memory_idx = self.logits_bank[head_id](head,enqueue=True if view_id==0 else False)
                    pseudo_label=self.sk[head_id](memory)[-batch_size:,:] if memory_idx==0 else self.sk[head_id](memory)[memory_idx-batch_size:memory_idx,:]
                    pseudo_labels_view.append(pseudo_label)
                pseudo_labels.append(pseudo_labels_view)
      
        self.i += 1
        total_loss=[]
        for i, (logits_head, label_head) in enumerate(zip(logits, pseudo_labels)):
            loss = 0  
            if self.label_quality_show:
                self.labels[i].append(label_head.cpu())  

            loss += self.criterion(logits_head, label_head)
            total_loss.append(loss)

        self.current_loss = sum(total_loss).item()

        if target is not None and self.label_quality_show:
            self.target.append(target.cpu())

        return total_loss, pseudo_labels[0]

    def single_forward(self, logits):
        pseudo_label = self.sk[0](logits)
        loss=self.criterion(logits,pseudo_label)
        return loss

    def reset(self):
        self.labels=[[] for _ in range(self.num_heads)]
        self.target=[]

    def prediction_log(self,top_rho=False):
        assert len(self.target)>0 and len(self.target) == len(self.labels[0])
        probs = [torch.cat(head,dim=0) for head in self.labels]
        predictions = [torch.argmax(head,dim=1) for head in probs]
        targets = torch.cat(self.target,dim=0)
        combine = [{'predictions': pred, 'probabilities': prob, 'targets': targets} for pred,prob in zip(predictions,probs)]

        if top_rho:
            select_num = int(targets.size(0) * self.sk[0].rho)
            print(f"top_rho select_num: {select_num}")
            sample_w = [torch.sum(head,dim=1) for head in probs]
            sample_top = [torch.topk(head, select_num, 0, largest=True)[1] for head in sample_w]
            pred_top = [torch.index_select(pred, 0, ind) for pred,ind in zip(predictions, sample_top)]
            prob_top = [torch.index_select(prob, 0, ind) for prob,ind in zip(probs, sample_top)]
            target_top = [torch.index_select(targets, 0, sample) for sample in sample_top]
            combine_top = [{'predictions': pred, 'probabilities': prob, 'targets': target_top[i]} for i,(pred,prob) in enumerate(zip(pred_top, prob_top))]
            
            return combine, combine_top
        else:
            return combine

    def set_gamma(self, gamma):
        for sk in self.sk:
            sk.gamma = gamma

    def set_rho(self, current, total):
        for sk in self.sk:
            if self.rho_strategy == "sigmoid":
                sk.rho = sigmoid_rampup(current, total)* self.rho_upper + self.rho_base
            elif self.rho_strategy == "linear":
                sk.rho = current / total * self.rho_upper + self.rho_base
            else:
                raise NotImplementedError

class ImprovedAdaptiveRho:
    def __init__(self, rho_0=0.1, rho_max=1.0, smoothing=0.9):
        self.rho_0 = rho_0
        self.rho_max = rho_max
        self.prev_loss = None
        self.max_loss_delta = 1e-6
        self.current_rho = rho_0
        self.smoothing = smoothing  

    def update_rho(self, current_loss):
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return self.rho_0

        loss_delta = max(self.prev_loss - current_loss, 0)  
        self.max_loss_delta = max(self.max_loss_delta, loss_delta)
        self.prev_loss = current_loss       
        progress = 1.0 - (loss_delta / self.max_loss_delta)
        target_rho = self.rho_0 + (self.rho_max - self.rho_0) * progress
        

        self.current_rho = self.smoothing * self.current_rho + (1 - self.smoothing) * target_rho
        return float(np.clip(self.current_rho, self.rho_0, self.rho_max))
