from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import normalized_mutual_info_score

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, acc, pur


def inference(loader, model, device, view, data_size):
    model.eval()
    commonZ_fused_high = []
    commonZ_fused_high_qhs = []
    labels_vector = []
    pred_vectors = []
    Xs = []
    Zs = []
    Hs = []
    Ss = []
    Qs = []
    for v in range(view):
        pred_vectors.append([])
        Xs.append([])
        Zs.append([])
        Hs.append([])
        Ss.append([])
        Qs.append([])
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            xrs, zs, hs, qls = model.forward(xs)
            commonz_fused_high, commonz_fused_high_qhs, _, _,ss = model.ViewFusion(xs)
            commonz_fused_high = commonz_fused_high.detach()
            commonz_fused_high_qhs = commonz_fused_high_qhs.detach()
            commonZ_fused_high.extend(commonz_fused_high.cpu().detach().numpy())
            commonZ_fused_high_qhs.extend(commonz_fused_high_qhs.cpu().detach().numpy())
        for v in range(view):
            zs[v] = zs[v].detach()
            hs[v] = hs[v].detach()
            Xs[v].extend(xs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
            Hs[v].extend(hs[v].cpu().detach().numpy())
            Ss[v].extend(ss[v].cpu().detach().numpy())
            Qs[v].extend(qls[v].cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    for v in range(view):
        Xs[v] = np.array(Xs[v])
        Zs[v] = np.array(Zs[v])
        Hs[v] = np.array(Hs[v])
        Ss[v] = np.array(Ss[v])
        Qs[v] = np.array(Qs[v])     
        pred_vectors[v] = np.array(pred_vectors[v])
    return labels_vector, Hs



def valid(model, device, dataset, view, data_size, class_num):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    labels_vector, h_vectors = inference(test_loader, model, device, view, data_size)
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    if len(labels_vector) > 10000:
        kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
    h = np.concatenate(h_vectors, axis=1)
    pseudo_label = kmeans.fit_predict(h)
    nmi, acc, pur = evaluate(labels_vector, pseudo_label) 
    return acc, nmi, pur

    
