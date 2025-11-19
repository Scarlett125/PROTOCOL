import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class  Reclass_loss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=1000):
        super(Reclass_loss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.weight = None

    def cal_weight_for_classes(self, labels):
        if labels.dim() == 2:
            labels = labels.squeeze()
        
        cls_num_list = torch.bincount(labels, minlength=self.num_classes)
        weight = cls_num_list / cls_num_list.sum()
        weight = weight.to(torch.device('cuda'))       
        return weight


    def forward(self, features, sup_logits):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        ss = features.shape[0]
        batch_size = self.K 
        labels = torch.argmax(sup_logits, dim=1).view(-1, 1).to(device) 
        weights = self.cal_weight_for_classes(labels)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device) 
        

        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature
        ) 
        

        anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(weights + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() 
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask


        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1) 

        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12) 

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)


        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
