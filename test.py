import torch
import argparse
import numpy as np
import random
import os
from network import PROTOCOL
from metric import valid
from dataloader import load_data
from config import create_config


def get_test_args(Dataname, IM_ratio):

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


    if args.dataset == "Hdigit":
        args.fine_tune_structure_epochs = 100
        args.fine_tune_IM_epochs = 100
        seed = 10

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(seed)
    
    p = create_config(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, dims, view, data_size, class_num = load_data(args.dataset, IM_ratio)
    
    return p, args, device, dataset, dims, view, data_size, class_num

def test_model(model_path, dataset, dims, view, data_size, class_num, device):

    model = PROTOCOL(view, 1, class_num, dims, 512, 128, device)
    model = model.to(device)   
    print(f"load model weight: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))    
    acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num)   
    return acc, nmi, pur

if __name__ == '__main__':

    IM_ratio = 0.1  
    Dataname = 'Hdigit'
    print(f"\n{'='*50}")
    print(f"test dataset: {Dataname}")
    print(f"{'='*50}")
    
    p, args, device, dataset, dims, view, data_size, class_num = get_test_args(Dataname, IM_ratio)          
    model_path = f'./models/{Dataname}_PROTOCOL.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: model file {model_path} does not exist!")
        exit()
        
    acc, nmi, pur = test_model(model_path, dataset, dims, view, data_size, class_num, device)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(acc, nmi, pur))    

            