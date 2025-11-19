import torch
from network import PROTOCOL
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from losses.loss import Feature_loss,Class_loss
from losses.losses_POT import POT_loss
from losses.losses_Reclass import Reclass_loss
from dataloader import load_data
import os
import pandas as pd
import torch.nn.functional as F
from mvc_ot.config import create_config


def get_args(Dataname,IM_ratio):
    print(Dataname)
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default=Dataname)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--temperature_l", default=1.0)
    parser.add_argument("--learning_rate", default=0.0003)
    parser.add_argument("--weight_decay", default=0.)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--rec_epochs", default=200)
    parser.add_argument("--fine_tune_IM_epochs", default=50)
    parser.add_argument("--fine_tune_alignment_epochs", default=100)
    parser.add_argument("--low_feature_dim", default=512)
    parser.add_argument("--high_feature_dim", default=128)
    parser.add_argument("--num_heads",default=1,type=int)
    parser.add_argument("--num_classes",default=[10],type=int,nargs="+")
    parser.add_argument("--output_dir", default="experiments/", type=str, help="output_dir")
    parser.add_argument('--setup', default="cluster")
    parser.add_argument("--epochs",default=50,type=int)
    parser.add_argument("--gamma_bound", default=0.1, type=float)
    parser.add_argument("--sk_factor", default=0.1, type=float)
    parser.add_argument("--sk_iter", default=3, type=int)
    parser.add_argument("--sk_iter_limit", default=1000, type=int)
    parser.add_argument("--rho_base",default=0.1,type=float)
    parser.add_argument("--rho_upper",default=1.0,type=float)
    parser.add_argument("--rho_fix", default=False, action='store_true', help='fix rho')
    parser.add_argument("--rho_strategy", default="sigmoid", type=str, help="sigmoid/linear")
    parser.add_argument("--label_quality_show", default=False, action='store_true', help='show pseudo label quality')
    parser.add_argument('--alpha', default=0.8, type=float, help='contrast weight among samples')
    parser.add_argument('--beta', default=0.2, type=float, help='contrast weight between centers and samples')
    parser.add_argument('--gamma', default=1.0, type=float, help='Rebalancedclass loss')

    args = parser.parse_args()
    p = create_config(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "Hdigit":
        args.fine_tune_alignment_epochs = 100
        args.fine_tune_IM_epochs = 100
        seed = 10
    

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(seed)

    dataset, dims, view, data_size, class_num= load_data(args.dataset,IM_ratio)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
    return p, args,device,dataset, dims, view, data_size, class_num,data_loader

def pre_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _,_ = model(xs)
        loss_list = []
        for v in range(view):
            rl = mse(xs[v], xrs[v])
            loss_list.append(rl)
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))



def fine_tune_Alignment(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs, qls = model(xs)
        commonz, commonz_qhs,S,weights, SV = model.ViewFusion(xs)

        sorted_indices = torch.argsort(weights, descending=False) 
        sorted_weights = sorted(weights, reverse=True)
        r = 2 / (view * (view - 1))
        half_views = len(sorted_indices) // 2
        if(len(sorted_indices) % 2 == 1): 
            num_views_to_update = half_views + 1
        else:
            num_views_to_update = half_views
        weight_big = sum(sorted_weights[:num_views_to_update])*r
        weight_small = (1 - sum(sorted_weights[:num_views_to_update]))*r
        loss_list = []
        for v in range(view):
            if sorted_indices[v] < num_views_to_update:                                  
                loss_list.append(weight_big * criterion_feature.Feature_Structure_Contrastive_Alignment(hs[v], commonz, S))
                loss_list.append(weight_big * criterion_class.Class_Alignment(qls[v], commonz_qhs))
                loss_list.append(mes(xs[v], xrs[v]))
                loss_list.append(weight_big *mes(zs[v], SV[v]))
            else:
                loss_list.append(weight_small * criterion_feature.Feature_Structure_Contrastive_Alignment(hs[v], commonz, S))
                loss_list.append(weight_small * criterion_class.Class_Alignment(qls[v], commonz_qhs))
                loss_list.append(mes(xs[v], xrs[v]))
                loss_list.append(weight_small *mes(zs[v], SV[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    return tot_loss/len(data_loader),weights



def fine_tune_Imbalanced(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs, qls = model(xs)      
        commonz, commonz_qhs,S, weights, SV = model.ViewFusion(xs)      
        a = 0.5 
        v = len(qls) 
        weighted_features = a * commonz_qhs
        for ql in qls:
            weighted_features += ((1-a)/v) * ql
        sk_loss_common, pseudo_label_common = criterion_POT([weighted_features], target=None, data_idxs=None)            
        sorted_indices = torch.argsort(weights, descending=False) 
        sorted_weights = sorted(weights, reverse=True)
        r = 2 / (view * (view - 1))
        half_views = len(sorted_indices) // 2
        if(len(sorted_indices) % 2 == 1): 
            num_views_to_update = half_views + 1
        else:
            num_views_to_update = half_views
        weight_big = sum(sorted_weights[:num_views_to_update])*r
        weight_small = (1 - sum(sorted_weights[:num_views_to_update]))*r
        loss_list = []
        sk_loss_list = []
        pseudo_label_v = []
        pseudo_label_w = []
        for v in range(view):
            if sorted_indices[v] < num_views_to_update:                
                loss_list.append(weight_big * criterion_feature.ReFeature_Structure_Contrastive(hs[v], commonz, S, pseudo_label_common)) 
                loss_list.append(weight_big * criterion_Reclass(qls[v], pseudo_label_common))
                loss_list.append(mes(xs[v], xrs[v]))
            else:               
                loss_list.append(weight_small * criterion_feature.ReFeature_Structure_Contrastive(hs[v], commonz, S, pseudo_label_common))
                loss_list.append(weight_small * criterion_Reclass(qls[v], pseudo_label_common))      
                loss_list.append(mes(xs[v], xrs[v]))
        sk_loss_list.append(sk_loss_common)
        sk_loss_list = [t[0].item() for t in sk_loss_list]
        sk_loss = sum(sk_loss_list)
        loss = sum(loss_list) 
        losses = sk_loss + loss
        losses.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    return tot_loss/len(data_loader), weights




    
if __name__ == '__main__':
    if not os.path.exists('./models'):
        os.makedirs('./models')
    IM_ratio = 0.1
    Dataname = 'Hdigit'
    p,args, device,dataset, dims, view, data_size, class_num,data_loader = get_args(Dataname,IM_ratio)    
    model = PROTOCOL(view, args.num_heads,class_num, dims, args.low_feature_dim, args.high_feature_dim, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion_feature = Feature_loss(args.batch_size, args.temperature_f, device,class_num).to(device)
    criterion_class = Class_loss(view,args.batch_size, class_num, args.temperature_l, device).to(device)
    criterion_POT = POT_loss(p,factor=p["sk_factor"],num_iter=p["sk_iter"],total_iter=len(data_loader) * p['epochs'], start_iter=0)
    criterion_Reclass = Reclass_loss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, temperature=args.temperature_f,  K = args.batch_size, num_classes=class_num).cuda()
   
    
    epoch = 1
    while epoch <= args.rec_epochs:
        pre_train(epoch)
        epoch += 1
        
    while epoch <= args.rec_epochs + args.fine_tune_alignment_epochs:
        fine_loss,weights = fine_tune_Alignment(epoch)
        epoch += 1

    while epoch <= args.rec_epochs  + args.fine_tune_IM_epochs + args.fine_tune_alignment_epochs:
        fine_loss,weights = fine_tune_Imbalanced(epoch)
        if epoch == args.rec_epochs  + args.fine_tune_IM_epochs + args.fine_tune_alignment_epochs:
            print("---------train over---------:",Dataname)
            print("Clustering results: " )
            acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num)
            print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(acc, nmi, pur))   
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset  +'.pth')
            print('Saving model...')
        epoch += 1

            
    
       


