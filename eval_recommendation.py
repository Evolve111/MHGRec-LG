import os
import sys
import numpy as np
import pickle
import scipy.sparse as sp
import logging
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from preprocess_recommendation import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor
from utils import convert_to_networkx_graph, temp_edge_type_lookup
import networkx as nx

from model_recommendation_eval import Model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Bookcrossing')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_hid', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset_seed', type=int, default=2)

# Multimodal item embeddings (optional)
parser.add_argument('--mm_dir', type=str, default='')
parser.add_argument('--mm_item_type', type=int, default=1)
parser.add_argument('--mm_fusion', type=str, default='weighted', choices=['sum','avg','weighted','concat','text','image'])
parser.add_argument('--mm_alpha', type=float, default=0.6)
parser.add_argument('--mm_norm', action='store_true', default=False)
parser.add_argument('--topk', type=int, default=10, help='Top-K for ranking metrics')
parser.add_argument('--eval_neg_per_pos', type=int, default=99, help='Number of negatives per positive for user-level eval')

# Explicit user multimodal preferences (optional)
parser.add_argument('--use_user_mm', action='store_true', default=False, help='Use explicit user multimodal preference profiles')
parser.add_argument('--user_type', type=int, default=0, help='node type id for users')
parser.add_argument('--user_mm_method', type=str, default='attn', choices=['attn','avg'], help='aggregation method for user profiles')
parser.add_argument('--user_query_dim', type=int, default=None, help='query dim for attention aggregator (defaults to item feature dim)')
parser.add_argument('--user_mm_norm', action='store_true', default=False, help='L2-normalize user profiles after aggregation')


def get_state_list(G, target):
    bfs_tree_result = nx.bfs_tree(G, target)
    bfs_node_order = [target]
    for edge in list(bfs_tree_result.edges()):
        bfs_node_order.append(edge[1])
    return bfs_node_order[::-1]


def get_connection_dict(G, state_list):
    connection_dict = dict()
    state_order_dict = dict(zip(state_list, np.arange(len(state_list))))
    for state in state_list:
        connection_dict[state] = [neighbor for neighbor in list(G.neighbors(state)) if state_order_dict[neighbor] < state_order_dict[state]]
    return connection_dict


def construct_arch(G, state_list, edge_type_lookup_dict):
    connection_dict = get_connection_dict(G, state_list)
    seq_arch, res_arch = [], []
    for i in range(1, len(state_list)):
        this_node = state_list[i]
        this_neighbors = connection_dict[this_node]
        if state_list[i-1] in this_neighbors:
            this_edge_type = G.nodes[this_node]['type'][0] + G.nodes[state_list[i-1]]['type'][0]
            seq_arch.append(edge_type_lookup_dict[this_edge_type])
        else:
            seq_arch.append(edge_type_lookup_dict['O'])
    for i in range(len(state_list)):
        this_node = state_list[i]
        for j in range(i+2, len(G.nodes)):
            if this_node in connection_dict[state_list[j]]:
                this_edge_type = G.nodes[state_list[j]]['type'][0] + G.nodes[this_node]['type'][0]
                try:
                    res_arch.append(edge_type_lookup_dict[this_edge_type])
                except Exception:
                    raise
            else:
                res_arch.append(edge_type_lookup_dict['O'])
    return (seq_arch, res_arch)


def structure2arch_eval(dataset_key, test_structure_list_sym=None):
    assert dataset_key.lower() in ('bookcrossing','amazons'), "Unsupported dataset"
    assert test_structure_list_sym is not None and len(test_structure_list_sym) > 0
    archs = {dataset_key: {'source': ([], []), 'target': ([], [])}}
    edge_type_lookup_dict = temp_edge_type_lookup(dataset_key.lower())
    for test_structure in test_structure_list_sym:
        G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
        state_list = get_state_list(G, 1)  # target node is B
        arch_target = construct_arch(G, state_list, edge_type_lookup_dict)
        state_list = get_state_list(G, 0)  # target node is U
        arch_source = construct_arch(G, state_list, edge_type_lookup_dict)
        archs[dataset_key]['source'][0].append(arch_source[0])
        archs[dataset_key]['source'][1].append(arch_source[1])
        archs[dataset_key]['target'][0].append(arch_target[0])
        archs[dataset_key]['target'][1].append(arch_target[1])
    return archs


def load_data(datadir, args, device):
    prefix = os.path.join(datadir, args.dataset)

    # node types
    node_types = np.load(os.path.join(prefix, 'node_types.npy'))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).to(device)

    # adjacency tensors (normalized)
    adjs_offset = pickle.load(open(os.path.join(prefix, 'adjs_offset.pkl'), 'rb'))
    adjs_pt = []
    if '0' in adjs_offset:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(adjs_offset['0'] + sp.eye(adjs_offset['0'].shape[0], dtype=np.float32))).to(device))
    for i in range(1, int(max(adjs_offset.keys())) + 1):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).to(device))
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).to(device))
    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()).to(device))
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape).to(device))

    # labels
    pos = np.load(os.path.join(prefix, f"pos_pairs_offset_larger_than_2_{args.dataset_seed}.npz"))
    pos_train = pos['train']
    pos_val = pos['val']
    pos_test = pos['test']

    neg = np.load(os.path.join(prefix, f"neg_pairs_offset_for_pos_larger_than_2_{args.dataset_seed}.npz"))
    neg_train = neg['train']
    neg_val = neg['val']
    neg_test = neg['test']

    # heterogeneous node features
    type_counts = [int((node_types == k).sum().item()) for k in range(num_node_types)]
    in_dims = type_counts.copy()
    node_feats = []

    fused_item_feats = None
    item_feature_dim = None
    if args.mm_dir and os.path.isdir(args.mm_dir):
        text_path = os.path.join(args.mm_dir, 'embed_text.npy')
        img_path = os.path.join(args.mm_dir, 'embed_image.npy')
        text_emb = np.load(text_path) if os.path.isfile(text_path) else None
        img_emb = np.load(img_path) if os.path.isfile(img_path) else None
        num_items = type_counts[args.mm_item_type]
        if args.mm_norm:
            if text_emb is not None:
                norms = np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-12
                text_emb = text_emb / norms
            if img_emb is not None:
                norms = np.linalg.norm(img_emb, axis=1, keepdims=True) + 1e-12
                img_emb = img_emb / norms
        if args.mm_fusion == 'sum':
            if text_emb is not None and img_emb is not None:
                fused_item_feats = torch.from_numpy(text_emb + img_emb).float().to(device)
        elif args.mm_fusion == 'avg':
            if text_emb is not None and img_emb is not None:
                fused_item_feats = torch.from_numpy((text_emb + img_emb) / 2.0).float().to(device)
        elif args.mm_fusion == 'weighted':
            if text_emb is not None and img_emb is not None:
                fused_item_feats = torch.from_numpy(args.mm_alpha * img_emb + (1 - args.mm_alpha) * text_emb).float().to(device)
        elif args.mm_fusion == 'concat':
            if text_emb is not None and img_emb is not None:
                fused_item_feats = torch.from_numpy(np.concatenate([text_emb, img_emb], axis=1)).float().to(device)
        elif args.mm_fusion == 'text' and text_emb is not None:
            fused_item_feats = torch.from_numpy(text_emb).float().to(device)
        elif args.mm_fusion == 'image' and img_emb is not None:
            fused_item_feats = torch.from_numpy(img_emb).float().to(device)

    # If multimodal item features are present, update in_dims for that node type
    if fused_item_feats is not None:
        item_feature_dim = fused_item_feats.shape[1]
        in_dims[args.mm_item_type] = item_feature_dim

    # Build user-item history map from training positives (global indices)
    if args.use_user_mm and fused_item_feats is not None:
        from user_profile_aggregator import UserPreferenceAggregator
        import torch.nn as nn

        node_types_tensor = node_types
        user_global_indices = torch.nonzero(node_types_tensor == args.user_type, as_tuple=False).squeeze(1).cpu().numpy().tolist()
        item_global_indices = torch.nonzero(node_types_tensor == args.mm_item_type, as_tuple=False).squeeze(1).cpu().numpy().tolist()
        user_global_to_local = {g: i for i, g in enumerate(user_global_indices)}
        item_global_to_local = {g: i for i, g in enumerate(item_global_indices)}

        user_history_local = {}
        for pair in pos_train:
            u_g = int(pair[0]); i_g = int(pair[1])
            if (u_g not in user_global_to_local) or (i_g not in item_global_to_local):
                continue
            u_l = user_global_to_local[u_g]
            i_l = item_global_to_local[i_g]
            user_history_local.setdefault(u_l, []).append(i_l)

        num_users = len(user_global_indices)
        query_dim = args.user_query_dim if args.user_query_dim is not None else item_feature_dim
        user_query_embedding = nn.Embedding(num_users, query_dim).to(device)

        if args.user_mm_method == 'attn':
            aggregator = UserPreferenceAggregator(in_dim=item_feature_dim, query_dim=query_dim).to(device)
            user_queries = user_query_embedding(torch.arange(num_users, device=device))
            all_item_local_indices = torch.arange(len(item_global_indices), device=device)
            user_mm_profiles = aggregator(user_queries, fused_item_feats, user_history_local, all_item_local_indices)
        else:
            user_mm_profiles = torch.zeros((num_users, item_feature_dim), device=device)
            for u_l, items_l in user_history_local.items():
                if len(items_l) == 0:
                    continue
                feats = fused_item_feats[torch.tensor(items_l, dtype=torch.long, device=device)]
                user_mm_profiles[u_l] = feats.mean(dim=0)
            if (len(user_history_local) < num_users):
                user_queries = user_query_embedding(torch.arange(num_users, device=device))
                proj = nn.Linear(query_dim, item_feature_dim).to(device)
                user_mm_profiles[user_mm_profiles.sum(dim=1) == 0] = proj(user_queries[user_mm_profiles.sum(dim=1) == 0])

        if args.user_mm_norm:
            norms = torch.norm(user_mm_profiles, dim=1, keepdim=True) + 1e-12
            user_mm_profiles = user_mm_profiles / norms

    for k in range(num_node_types):
        if fused_item_feats is not None and k == args.mm_item_type:
            node_feats.append(fused_item_feats)
        elif args.use_user_mm and k == args.user_type and fused_item_feats is not None:
            node_feats.append(user_mm_profiles)
        else:
            count_k = type_counts[k]
            i = torch.stack((torch.arange(count_k, dtype=torch.long), torch.arange(count_k, dtype=torch.long)))
            v = torch.ones(count_k)
            node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([count_k, count_k])).to(device))

    return node_types, adjs_pt, pos_train, pos_val, pos_test, neg_train, neg_val, neg_test, in_dims, node_feats


def infer_eval(dataset_key, archs, node_feats, node_types, adjs, pos_val, neg_val, pos_test, neg_test, model_s, model_t, connection_dict_s, connection_dict_t):
    model_s.eval()
    model_t.eval()
    with torch.no_grad():
        out_s = model_s(node_feats, node_types, adjs, archs[dataset_key]['source'][0], archs[dataset_key]['source'][1], connection_dict_s)
        out_t = model_t(node_feats, node_types, adjs, archs[dataset_key]['target'][0], archs[dataset_key]['target'][1], connection_dict_t)

    pos_val_prod = torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)
    neg_val_prod = torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)
    loss = - torch.mean(F.logsigmoid(pos_val_prod) + F.logsigmoid(- neg_val_prod))

    y_true_val = np.zeros((pos_val.shape[0] + neg_val.shape[0]), dtype=np.long)
    y_true_val[:pos_val.shape[0]] = 1
    y_pred_val = np.concatenate((torch.sigmoid(pos_val_prod).cpu().numpy(), torch.sigmoid(neg_val_prod).cpu().numpy()))
    auc_val = roc_auc_score(y_true_val, y_pred_val)

    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)

    y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.long)
    y_true_test[:pos_test.shape[0]] = 1
    y_pred_test = np.concatenate((torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))
    auc_test = roc_auc_score(y_true_test, y_pred_test)

    # 用户级 Top-K 指标：每个用户 1 正 + N 负
    def compute_user_topk_metrics(pos_pairs, neg_pairs, out_s, out_t, topk: int, neg_per_pos: int, node_types_arr, item_type_id, neg_global_arr=None):
        import random
        random.seed(args.seed)
        from collections import defaultdict

        user_pos = defaultdict(list)
        for u, i in pos_pairs:
            user_pos[int(u)].append(int(i))

        user_negs = defaultdict(list)
        for u, i in neg_pairs:
            user_negs[int(u)].append(int(i))

        item_indices = np.where(node_types_arr == item_type_id)[0].tolist()
        users = list(user_pos.keys())
        hits = 0
        precision_sum = 0.0
        dcg_sum = 0.0
        idcg_sum = float(len(users))

        for u in users:
            pos_items = user_pos[u]
            if len(pos_items) == 0:
                continue
            pos_item = pos_items[0]

            negs_u = user_negs[u][:]
            if neg_global_arr is not None:
                mask = (neg_global_arr[:,0] == u)
                negs_u.extend([int(x) for x in neg_global_arr[mask][:,1].tolist()])

            negs_u = list({i for i in negs_u if i != pos_item})
            if len(negs_u) < neg_per_pos:
                extras = [i for i in item_indices if i != pos_item and i not in negs_u]
                if len(extras) > 0:
                    random.shuffle(extras)
                    take = min(neg_per_pos - len(negs_u), len(extras))
                    negs_u.extend(extras[:take])

            if len(negs_u) > neg_per_pos:
                random.shuffle(negs_u)
                negs_u = negs_u[:neg_per_pos]

            cand_items = [pos_item] + negs_u
            scores = (out_s[u].unsqueeze(0) * out_t[cand_items]).sum(dim=-1)
            topk_use = min(topk, len(cand_items))
            topk_indices = torch.topk(scores, topk_use).indices.tolist()
            topk_items = [cand_items[idx] for idx in topk_indices]
            hit = 1 if pos_item in topk_items else 0
            hits += hit
            precision_sum += hit / float(topk_use)
            sorted_idx = torch.argsort(scores, descending=True).tolist()
            rank_pos = sorted_idx.index(cand_items.index(pos_item)) + 1
            dcg_sum += 1.0 / np.log2(rank_pos + 1)

        users_count = len(users)
        hr = hits / float(users_count) if users_count > 0 else 0.0
        recall = hr
        precision = precision_sum / float(users_count) if users_count > 0 else 0.0
        ndcg = dcg_sum / float(idcg_sum) if users_count > 0 else 0.0
        return recall, precision, ndcg, hr

    # 加载全局负样本池（若存在）
    prefix = os.path.join('data_recommendation', args.dataset)
    neg_global_path = os.path.join(prefix, 'neg_ratings_offset_smaller_than_3.npy')
    neg_global = None
    if os.path.isfile(neg_global_path):
        try:
            neg_global = np.load(neg_global_path)
        except Exception:
            neg_global = None

    node_types_arr = node_types.cpu().numpy()
    recall10, precision10, ndcg10, hr10 = compute_user_topk_metrics(
        pos_test, neg_test, out_s, out_t, args.topk, args.eval_neg_per_pos, node_types_arr, args.mm_item_type,
        neg_global_arr=neg_global
    )
    logging.info(f"Test AUC {auc_test:.4f} | Top-{args.topk} HR {hr10:.4f} | Recall {recall10:.4f} | Precision {precision10:.4f} | NDCG {ndcg10:.4f}")
    logging.info(f"Val AUC {auc_val:.4f} | Val loss {loss.item():.6f}")
    return loss.item(), auc_val, auc_test, {'hr@10': hr10, 'recall@10': recall10, 'precision@10': precision10, 'ndcg@10': ndcg10}


def main():
    args = parser.parse_args()
    assert args.dataset.lower() == 'bookcrossing', "Only 'Bookcrossing' is supported"

    # logging
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

    # device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # data
    node_types, adjs_pt, pos_train, pos_val, pos_test, neg_train, neg_val, neg_test, in_dims, node_feats = load_data(datadir='data_recommendation', args=args, device=device)

    # default Bookcrossing gene pools (paths)
    BOOK_UBU = [['U', 'B', 'U'], [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]]
    BOOK_UBUB = [['U', 'B', 'U', 'B'], [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ]]
    BOOK_UBUB_CYCLE = [['U', 'B', 'U', 'B'], [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]]
    BOOK_UBUBU = [['U', 'B', 'U', 'B', 'U'], [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ]]
    BOOK_UBUBUB = [['U', 'B', 'U', 'B', 'U', 'B'], [
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0]
    ]]
    gene_pools = [BOOK_UBU, BOOK_UBUB, BOOK_UBUB_CYCLE, BOOK_UBUBU, BOOK_UBUBUB]

    archs = structure2arch_eval(args.dataset, test_structure_list_sym=gene_pools)
    steps_s = [len(meta) for meta in archs[args.dataset]['source'][0]]
    steps_t = [len(meta) for meta in archs[args.dataset]['target'][0]]

    model_s = Model(in_dims, args.n_hid, steps_s, dropout=args.dropout).to(device)
    model_t = Model(in_dims, args.n_hid, steps_t, dropout=args.dropout).to(device)

    connection_dict_s = {}
    connection_dict_t = {}

    val_loss, auc_val, auc_test, top10 = infer_eval(
        args.dataset,
        archs,
        node_feats,
        node_types,
        adjs_pt,
        pos_val,
        neg_val,
        pos_test,
        neg_test,
        model_s,
        model_t,
        connection_dict_s,
        connection_dict_t,
    )

    print(f"Val AUC: {auc_val:.6f}")
    print(f"Test AUC: {auc_test:.6f}")
    print(f"Top-10 HR: {top10['hr@10']:.6f}, Recall: {top10['recall@10']:.6f}, Precision: {top10['precision@10']:.6f}, NDCG: {top10['ndcg@10']:.6f}")


if __name__ == '__main__':
    main()
