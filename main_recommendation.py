import os
import sys
import numpy as np
import pickle
import json
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import time
from datetime import datetime
import argparse
import logging
from model_recommendation import Model
import copy
from utils import *
import networkx as nx
import pdb
from llm_component_no_prob import LLM4Meta 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# numpy>=2.0 移除了 np.VisibleDeprecationWarning；做兼容处理
try:
    from numpy.exceptions import VisibleDeprecationWarning as NPVisibleDeprecationWarning
    warnings.filterwarnings("ignore", category=NPVisibleDeprecationWarning)
except Exception:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='Bookcrossing')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs') 
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--non_symmetric', default=False, action='store_true')
parser.add_argument('--test_known_metas', default=False, action='store_true')
parser.add_argument('--num_generations', type=int, default=20)
# Backward compatibility aliases
parser.add_argument('--num_iters', type=int, dest='num_generations', help='alias of --num_generations')
parser.add_argument('--neg_train_size', type=int, default=4)
parser.add_argument('--neg_val_test_size', type=int, default=100)
parser.add_argument('--loss_margin', type=float, default=0.3)
parser.add_argument('--dataset_seed', type=int, default=2)
parser.add_argument('--mm_dir', type=str, default='', help='path to multimodal dataset folder (e.g., 多模态数据集/bookcrossing-vit_bert)')
parser.add_argument('--multimodal_dir', type=str, dest='mm_dir', help='alias of --mm_dir')
parser.add_argument('--mm_item_type', type=int, default=1, help='node type id for items to use multimodal embeddings')
parser.add_argument('--mm_fusion', type=str, default='avg', help='fusion method: sum|avg|weighted|concat|text|image')
parser.add_argument('--fusion_method', type=str, dest='mm_fusion', help='alias of --mm_fusion')
parser.add_argument('--mm_alpha', type=float, default=0.5, help='alpha for weighted fusion (alpha*image + (1-alpha)*text)')
parser.add_argument('--mm_norm', default=True, action='store_true')
parser.add_argument('--trial', type=str, default=None, help='unused legacy flag; accepted for backward compatibility')
parser.add_argument('--full_item_eval', default=False, action='store_true', help='compute P@k/R@k by ranking over all items as candidates')
parser.add_argument('--random_search', default=False, action='store_true', help='ablation: disable LLM predictor & selector, use random candidate selection')
parser.add_argument('--no_grammar_translator', default=False, action='store_true', help='ablation: feed raw graph structure to LLM instead of grammar logic')
parser.add_argument('--disable_mm', default=False, action='store_true', help='disable multimodal features during search (use raw one-hot)')
parser.add_argument('--user_mm_precomputed', type=str, default='', help='path to precomputed user profiles (.npy) used in search')
parser.add_argument('--user_type', type=int, default=0, help='node type id for users')
parser.add_argument('--delta', type=float, default=0.8, help='similarity threshold for path equivalence')
parser.add_argument('--w1', type=float, default=0.5, help='weight for predicted performance')
parser.add_argument('--w2', type=float, default=0.3, help='weight for semantic similarity prior')
parser.add_argument('--w3', type=float, default=0.2, help='weight for calibration confidence')
parser.add_argument('--population_size', type=int, default=10, help='number of individuals in the population')
parser.add_argument('--patience', type=int, default=30, help='early stopping patience during evaluation training')
parser.add_argument('--llm_temperature', type=float, default=0.6)
parser.add_argument('--llm_top_p', type=float, default=1.0)
parser.add_argument('--llm_max_tokens', type=int, default=1000)
parser.add_argument('--llm_prompt_price_per_1k', type=float, default=0.0)
parser.add_argument('--llm_completion_price_per_1k', type=float, default=0.0)
parser.add_argument('--prompt_format', type=str, default='A', choices=['A','B','C','D'])
parser.add_argument('--few_shot_k', type=int, default=10)
args = parser.parse_args()

# Use local Ollama; pass None client to LLM4Meta which uses utils.get_gpt_completion
client = None

# Device setup: use CUDA if available; otherwise fall back to CPU
device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

#########################################################################

if(args.test_known_metas):
    POPULATION_SIZE = 1
else:
    POPULATION_SIZE = int(args.population_size)
GENE_NUM = 1  # Number of genes carried by each individual
GENE_POOL_SIZE = POPULATION_SIZE * GENE_NUM
ELIMINATE_RATE = 0.4 

current_date = datetime.now().date().strftime("%Y-%m-%d")

##############################################################################################################

def train(archs, node_feats, node_types, adjs, pos_train, neg_train, model_s, model_t, connection_dict_s, connection_dict_t, optimizer):

    model_s.train()
    model_t.train()
    optimizer.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1], connection_dict_s)
    out_t = model_t(node_feats, node_types, adjs, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1], connection_dict_t)
    
    loss = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss.backward()
    optimizer.step()
    return loss.item()

def infer(archs, node_feats, node_types, adjs, pos_val, neg_val, pos_test, neg_test, model_s, model_t, connection_dict_s, connection_dict_t):

    model_s.eval()
    model_t.eval()
    with torch.no_grad():
        out_s = model_s(node_feats, node_types, adjs, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1], connection_dict_s)
        out_t = model_t(node_feats, node_types, adjs, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1], connection_dict_t)
        # sanitize NaNs/Infs to keep metrics stable
        out_s = torch.nan_to_num(out_s, nan=0.0, posinf=0.0, neginf=0.0)
        out_t = torch.nan_to_num(out_t, nan=0.0, posinf=0.0, neginf=0.0)
    
    #* validation performance
    pos_val_prod = torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)
    neg_val_prod = torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)
    loss = - torch.mean(F.logsigmoid(pos_val_prod) + F.logsigmoid(- neg_val_prod))

    y_true_val = np.zeros((pos_val.shape[0] + neg_val.shape[0]), dtype=np.long)
    y_true_val[:pos_val.shape[0]] = 1
    y_pred_val = np.concatenate((torch.sigmoid(pos_val_prod).cpu().numpy(), torch.sigmoid(neg_val_prod).cpu().numpy()))
    # guard against NaNs/Infs
    y_pred_val = np.nan_to_num(y_pred_val, nan=0.5, posinf=1.0, neginf=0.0)
    auc_val = roc_auc_score(y_true_val, y_pred_val)

    #* test performance
    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)

    y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.long)
    y_true_test[:pos_test.shape[0]] = 1
    y_pred_test = np.concatenate((torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))
    y_pred_test = np.nan_to_num(y_pred_test, nan=0.5, posinf=1.0, neginf=0.0)
    auc_test = roc_auc_score(y_true_test, y_pred_test)
    
    def compute_topk_metrics(pos_pairs, neg_pairs, out_s, out_t, k: int = 10):
        from collections import defaultdict
        user_pos = defaultdict(list)
        user_cand = defaultdict(set)
        for u, i in pos_pairs:
            user_pos[int(u)].append(int(i))
            user_cand[int(u)].add(int(i))
        for u, i in neg_pairs:
            user_cand[int(u)].add(int(i))

        hits_total = 0
        precision_total = 0.0
        dcg_total = 0.0
        idcg_total = 0.0
        users_count = 0
        users_hit = 0
        pos_total = 0

        for u, cand_set in user_cand.items():
            pos_items = user_pos.get(u, [])
            if len(pos_items) == 0:
                continue
            pos_total += len(pos_items)
            items = list(cand_set)
            # Score all candidate items for user u
            scores = (out_s[u].unsqueeze(0) * out_t[items]).sum(dim=-1)
            topk = min(k, len(items))
            topk_indices = torch.topk(scores, topk).indices.tolist()
            topk_items = [items[idx] for idx in topk_indices]

            user_hits = len(set(pos_items) & set(topk_items))
            hits_total += user_hits
            if user_hits > 0:
                users_hit += 1
            precision_total += user_hits / float(topk)

            # NDCG@k
            sorted_indices = torch.argsort(scores, descending=True).tolist()
            item2rank = {items[idx]: (r + 1) for r, idx in enumerate(sorted_indices)}  # ranks start at 1
            dcg = 0.0
            for item in pos_items:
                rank = item2rank.get(item, None)
                if rank is not None and rank <= k:
                    dcg += 1.0 / np.log2(rank + 1)
            p_k = min(len(pos_items), k)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(p_k))  # ideal ranks 1..p_k
            dcg_total += dcg
            idcg_total += idcg
            users_count += 1

        recall_at_k = hits_total / pos_total if pos_total > 0 else 0.0
        precision_at_k = precision_total / users_count if users_count > 0 else 0.0
        ndcg_at_k = (dcg_total / idcg_total) if idcg_total > 0 else 0.0
        hitrate_at_k = users_hit / users_count if users_count > 0 else 0.0
        return recall_at_k, precision_at_k, ndcg_at_k, hitrate_at_k

    try:
        recall10, precision10, ndcg10, hr10 = compute_topk_metrics(pos_test, neg_test, out_s, out_t, k=10)
        recall20, precision20, ndcg20, hr20 = compute_topk_metrics(pos_test, neg_test, out_s, out_t, k=20)
        logging.info(f"SampledCandidates@10: HR {hr10:.4f} | Recall {recall10:.4f} | Precision {precision10:.4f} | NDCG {ndcg10:.4f}")
        logging.info(f"SampledCandidates@20: HR {hr20:.4f} | Recall {recall20:.4f} | Precision {precision20:.4f} | NDCG {ndcg20:.4f}")

        if args.full_item_eval:
            from collections import defaultdict
            user_pos = defaultdict(list)
            for u, i in pos_test:
                user_pos[int(u)].append(int(i))
            # all items indices (type id==1 by default)
            item_mask = (node_types.cpu().numpy() == args.mm_item_type)
            all_items = np.where(item_mask)[0].tolist()
            hits_total_10 = hits_total_20 = 0
            precision_total_10 = precision_total_20 = 0.0
            pos_total = 0
            users_count = 0
            for u, pos_items in user_pos.items():
                if len(pos_items) == 0:
                    continue
                pos_total += len(pos_items)
                scores = (out_s[u].unsqueeze(0) * out_t[all_items]).sum(dim=-1)
                top10_idx = torch.topk(scores, min(10, len(all_items))).indices.tolist()
                top20_idx = torch.topk(scores, min(20, len(all_items))).indices.tolist()
                top10_items = [all_items[idx] for idx in top10_idx]
                top20_items = [all_items[idx] for idx in top20_idx]
                hits10 = len(set(pos_items) & set(top10_items))
                hits20 = len(set(pos_items) & set(top20_items))
                hits_total_10 += hits10
                hits_total_20 += hits20
                precision_total_10 += hits10 / float(min(10, len(all_items)))
                precision_total_20 += hits20 / float(min(20, len(all_items)))
                users_count += 1
            recall10_full = hits_total_10 / pos_total if pos_total > 0 else 0.0
            recall20_full = hits_total_20 / pos_total if pos_total > 0 else 0.0
            precision10_full = precision_total_10 / users_count if users_count > 0 else 0.0
            precision20_full = precision_total_20 / users_count if users_count > 0 else 0.0
            logging.info(f"FullItems@10: Recall {recall10_full:.4f} | Precision {precision10_full:.4f}")
            logging.info(f"FullItems@20: Recall {recall20_full:.4f} | Precision {precision20_full:.4f}")
    except Exception as e:
        logging.warning(f"Top-k metric computation failed: {e}")
    
    return loss.item(), auc_val, auc_test

def get_state_list(G, target):
    # Generate BFS tree by searching reversely from the target node
    bfs_tree_result = nx.bfs_tree(G, target)
    bfs_node_order = [target]
    for edge in list(bfs_tree_result.edges()):
        bfs_node_order.append(edge[1])
    return bfs_node_order[::-1] # target node ranks last

def construct_arch(G, state_list, edge_type_lookup_dict):
    connection_dict = get_connection_dict(G, state_list)
    seq_arch, res_arch = [], []
    for i in range(1, len(state_list)):
        this_node = state_list[i]
        this_neighbors = connection_dict[this_node]     
        if(state_list[i-1] in this_neighbors):
            this_edge_type = G.nodes[this_node]['type'][0] + G.nodes[state_list[i-1]]['type'][0]
            seq_arch.append(edge_type_lookup_dict[this_edge_type])
        else:
            seq_arch.append(edge_type_lookup_dict['O'])
    for i in range(len(state_list)):
        this_node = state_list[i]
        for j in range(i+2, len(G.nodes)):
            if(this_node in connection_dict[state_list[j]]):
                this_edge_type = G.nodes[state_list[j]]['type'][0] + G.nodes[this_node]['type'][0]
                try:
                    res_arch.append(edge_type_lookup_dict.get(this_edge_type, edge_type_lookup_dict['O']))
                except Exception as e:
                    logging.warning(f"Unknown edge type {this_edge_type}; defaulting to 'O'. Error: {e}")
                    res_arch.append(edge_type_lookup_dict['O'])
            else:
                res_arch.append(edge_type_lookup_dict['O'])
    meta_arch = (seq_arch, res_arch)
    return meta_arch

def get_connection_dict(G, state_list):
    connection_dict = dict()
    state_order_dict = dict(zip(state_list, np.arange(len(state_list))))
    for state in state_list:
        connection_dict[state] = [neighbor for neighbor in list(G.neighbors(state)) if state_order_dict[neighbor] < state_order_dict[state]]
    return connection_dict


def load_data(datadir):
    prefix = os.path.join(datadir, args.dataset)

    #* load data
    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).to(device)

    adjs_offset = pickle.load(open(os.path.join(prefix, "adjs_offset.pkl"), "rb"))
    adjs_pt = []
    if '0' in adjs_offset:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(adjs_offset['0'] + sp.eye(adjs_offset['0'].shape[0], dtype=np.float32))).to(device))
    for i in range(1, int(max(adjs_offset.keys())) + 1):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).to(device))
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).to(device))
    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()).to(device))
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape).to(device))
    print("Loading {} adjs...".format(len(adjs_pt)))

    #* load labels
    pos = np.load(os.path.join(prefix, f"pos_pairs_offset_larger_than_2_{args.dataset_seed}.npz"))
    pos_train = pos['train']
    pos_val = pos['val']
    pos_test = pos['test']
    print(pos_train.shape, pos_val.shape, pos_test.shape)

    neg = np.load(os.path.join(prefix, f"neg_pairs_offset_for_pos_larger_than_2_{args.dataset_seed}.npz"))
    neg_train = neg['train']
    neg_val = neg['val']
    neg_test = neg['test']
    print(neg_train.shape, neg_val.shape, neg_test.shape)

    #* build heterogeneous node features: users one-hot; items multimodal embeddings if provided
    type_counts = [int((node_types == k).sum().item()) for k in range(num_node_types)]
    in_dims = type_counts.copy()
    node_feats = []

    fused_item_feats = None
    if (not args.disable_mm) and args.mm_dir and os.path.isdir(args.mm_dir):
        # Try to load multimodal embeddings
        text_path = os.path.join(args.mm_dir, 'embed_text.npy')
        img_path = os.path.join(args.mm_dir, 'embed_image.npy')
        text_emb = None
        img_emb = None
        if os.path.isfile(text_path):
            text_emb = np.load(text_path)
        if os.path.isfile(img_path):
            img_emb = np.load(img_path)

        num_items = type_counts[args.mm_item_type]
        if text_emb is not None and text_emb.shape[0] != num_items:
            logging.warning(f"Text embeddings rows ({text_emb.shape[0]}) != num_items ({num_items}). Proceeding may misalign features.")
        if img_emb is not None and img_emb.shape[0] != num_items:
            logging.warning(f"Image embeddings rows ({img_emb.shape[0]}) != num_items ({num_items}). Proceeding may misalign features.")

        # L2 normalize per row if requested
        if args.mm_norm:
            if text_emb is not None:
                norms = np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-12
                text_emb = text_emb / norms
            if img_emb is not None:
                norms = np.linalg.norm(img_emb, axis=1, keepdims=True) + 1e-12
                img_emb = img_emb / norms

        if text_emb is None and img_emb is None:
            logging.info("No multimodal embeddings found in mm_dir; falling back to one-hot for items.")
        else:
            # Fuse
            if img_emb is not None and text_emb is not None:
                if args.mm_fusion == 'sum':
                    fused = img_emb + text_emb
                elif args.mm_fusion == 'avg':
                    fused = (img_emb + text_emb) / 2.0
                elif args.mm_fusion == 'weighted':
                    fused = args.mm_alpha * img_emb + (1.0 - args.mm_alpha) * text_emb
                elif args.mm_fusion == 'concat':
                    fused = np.concatenate([img_emb, text_emb], axis=1)
                elif args.mm_fusion == 'image':
                    fused = img_emb
                elif args.mm_fusion == 'text':
                    fused = text_emb
                else:
                    logging.warning(f"Unknown fusion '{args.mm_fusion}', defaulting to avg.")
                    fused = (img_emb + text_emb) / 2.0
            else:
                fused = text_emb if text_emb is not None else img_emb

            fused_item_feats = torch.from_numpy(fused).float().to(device)
            in_dims[args.mm_item_type] = fused_item_feats.size(1)

    # optional precomputed user profiles for search
    user_mm_profiles = None
    if args.user_mm_precomputed and os.path.isfile(args.user_mm_precomputed):
        node_types_tensor = node_types if torch.is_tensor(node_types) else torch.from_numpy(node_types).to(device)
        user_global_indices = torch.nonzero(node_types_tensor == args.user_type, as_tuple=False).squeeze(1).cpu().numpy().tolist()
        precomp = np.load(args.user_mm_precomputed)
        assert precomp.shape[0] == len(user_global_indices), f"Precomputed profiles rows ({precomp.shape[0]}) != num_users ({len(user_global_indices)})"
        user_mm_profiles = torch.from_numpy(precomp).float().to(device)
        in_dims[args.user_type] = user_mm_profiles.size(1)

    for k in range(num_node_types):
        if fused_item_feats is not None and k == args.mm_item_type:
            node_feats.append(fused_item_feats)
        elif user_mm_profiles is not None and k == args.user_type:
            node_feats.append(user_mm_profiles)
        else:
            count_k = type_counts[k]
            i = torch.stack((torch.arange(count_k, dtype=torch.long), torch.arange(count_k, dtype=torch.long)))
            v = torch.ones(count_k)
            node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([count_k, count_k])).to(device))

    assert(len(in_dims) == len(node_feats))
    
    return node_types, num_node_types, adjs_pt, pos_train, pos_val, pos_test, neg_train, neg_val, neg_test, in_dims, node_feats

def structure2arch(test_structure_list_sym=[], test_structure_list_source=[], test_structure_list_target=[]):
    if(args.non_symmetric):
        assert (len(test_structure_list_source)>0 & len(test_structure_list_target)>0)
    else:
        assert len(test_structure_list_sym)>0
    
    archs = {args.dataset: {'source': ([],[]), 'target': ([],[])}} # Initialization
    edge_type_lookup_dict = temp_edge_type_lookup(dataset_string)
    if(not args.non_symmetric):
        for test_structure in test_structure_list_sym:
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_target = construct_arch(G, state_list, edge_type_lookup_dict) # target node is B
            state_list = get_state_list(G, 0)
            arch_source = construct_arch(G, state_list, edge_type_lookup_dict) # target node is U
            archs[args.dataset]['source'][0].append(arch_source[0])
            archs[args.dataset]['source'][1].append(arch_source[1])
            archs[args.dataset]['target'][0].append(arch_target[0])
            archs[args.dataset]['target'][1].append(arch_target[1])
    else:
        for test_structure in test_structure_list_source:
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_source = construct_arch(G, state_list, edge_type_lookup_dict) # target node is B
            archs[args.dataset]['source'][0].append(arch_source[0])
            archs[args.dataset]['source'][1].append(arch_source[1])
        for test_structure in test_structure_list_target:   
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_target = construct_arch(G, state_list, edge_type_lookup_dict) # target node is U
            archs[args.dataset]['target'][0].append(arch_target[0])
            archs[args.dataset]['target'][1].append(arch_target[1])
    return archs

def evaluate(gene_pools, dataset_string): 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    archs = structure2arch(gene_pools)

    steps_s = [len(meta) for meta in archs[args.dataset]["source"][0]] # steps_s: [4]
    steps_t = [len(meta) for meta in archs[args.dataset]["target"][0]] # steps_t: [6]

    model_s = Model(in_dims, args.n_hid, steps_s, dropout = args.dropout).to(device)
    model_t = Model(in_dims, args.n_hid, steps_t, dropout = args.dropout).to(device)

    connection_dict_s = {}
    connection_dict_t = {}

    optimizer = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )

    best_val = None
    final = None
    anchor = None
    # Convert indices to torch tensors on the correct device for safe indexing
    pos_train_t = torch.as_tensor(pos_train, dtype=torch.long, device=device)
    neg_train_t = torch.as_tensor(neg_train, dtype=torch.long, device=device)
    pos_val_t = torch.as_tensor(pos_val, dtype=torch.long, device=device)
    neg_val_t = torch.as_tensor(neg_val, dtype=torch.long, device=device)
    pos_test_t = torch.as_tensor(pos_test, dtype=torch.long, device=device)
    neg_test_t = torch.as_tensor(neg_test, dtype=torch.long, device=device)

    no_improve = 0
    for epoch in range(args.epochs):
        train_loss = train(archs, node_feats, node_types, adjs_pt, pos_train_t, neg_train_t, model_s, model_t, connection_dict_s, connection_dict_t, optimizer)
        val_loss, auc_val, auc_test = infer(archs, node_feats, node_types, adjs_pt, pos_val_t, neg_val_t, pos_test_t, neg_test_t, model_s, model_t, connection_dict_s, connection_dict_t)
        if(epoch%50==0):
            logging.info("Epoch {}; Train err {}; Val err {}; Val auc {}".format(epoch + 1, train_loss, val_loss, auc_val))
        if best_val is None or auc_val > best_val:
            best_val = auc_val
            final = auc_test
            anchor = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break
    logging.info("Best val auc {} at epoch {}; Test auc {}".format(best_val, anchor, final))
    
    return best_val, final

def eliminate_and_reproduce(old_gene_pools, population_performance):
    old_gene_pools_cp = copy.deepcopy(old_gene_pools)

    # Eliminate
    ranking = np.argsort(population_performance) # hit_rate_50 (HR20), refer to metrics()
    preserved_index = ranking[int(ELIMINATE_RATE*POPULATION_SIZE):]
    new_gene_pools = []
    for i in preserved_index:
        new_gene_pools = new_gene_pools + old_gene_pools_cp[GENE_NUM*i:(GENE_NUM*i+GENE_NUM)]

    # Reproduce
    while len(new_gene_pools) < len(old_gene_pools_cp):
        try:
            pre_p = np.exp(population_performance[preserved_index])
            pre_p = pre_p / pre_p.sum()
        except Exception as e:
            logging.warning(f"Failed to compute reproduction probabilities: {e}. Using uniform distribution.")
            if preserved_index.size == 0:
                preserved_index = np.arange(len(old_gene_pools_cp)//GENE_NUM)
            pre_p = np.ones(len(preserved_index), dtype=float) / len(preserved_index)
        index = np.random.choice(preserved_index, size = 1, p = pre_p)[0]
        new_gene_pools = new_gene_pools + old_gene_pools_cp[GENE_NUM*index:(GENE_NUM*index+GENE_NUM)]
    
    return new_gene_pools

def get_recent_performances(perf_dict, num_to_get):
    num_gens = len(perf_dict)
    if(num_gens>num_to_get):
        return {i : perf_dict[i] for i in range(num_gens-num_to_get, num_gens)}
    else:
        return perf_dict

def structure2logic(meta, dataset_string):
    nx_meta = convert_to_networkx_graph({'nodes': meta[0], 'edges': meta[1]})
    logic = graph2logic(nx_meta, dataset_string, initialization=True)
    return logic

if __name__ == "__main__":

    prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)
    if args.random_search:
        prefix += "_rs"
    if args.no_grammar_translator:
        prefix += "_nogram"
    prefix += f"_pf{args.prompt_format}_fs{args.few_shot_k}_seed{args.seed}"

    logdir_base = os.path.join(f"log_recommendation/train_threshold_2_delta_{args.delta}_datasetseed_{args.dataset_seed}_changeinit", args.dataset)
    os.makedirs(logdir_base, exist_ok=True)
    logdir = os.path.join(logdir_base, f"pf_{args.prompt_format}", f"fs_{args.few_shot_k}", f"seed_{args.seed}")
    # Ensure logdir exists; do not break if it already exists
    os.makedirs(logdir, exist_ok=True)
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    fh = logging.FileHandler(os.path.join(logdir, prefix + ".txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    dataset_string = args.dataset.lower()
    print('dataset_string: ', dataset_string)
    task_string = 'recommendation'
    dataset_task_string = dataset_string + '_' + task_string

    # Load data
    node_types, num_node_types, adjs_pt, pos_train, pos_val, pos_test, neg_train, neg_val, neg_test, in_dims, node_feats = load_data(datadir='data_recommendation')

    # cuda settings
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Initialization: Hyperparams
    gene_pools_history_dict = dict() 

    ##################################################################################################################
    # Initialization

    # LLM intialization
    trial='_component_v3l' 
    dialogs_save_path = logdir 
    llm4meta = LLM4Meta(client=client, dataset=dataset_string, downstream_task=task_string, dialogs_save_path=dialogs_save_path, random_search=args.random_search, no_grammar_translator=args.no_grammar_translator, delta=args.delta, w1=args.w1, w2=args.w2, w3=args.w3, llm_temperature=args.llm_temperature, llm_top_p=args.llm_top_p, llm_max_tokens=args.llm_max_tokens, llm_prompt_price_per_1k=args.llm_prompt_price_per_1k, llm_completion_price_per_1k=args.llm_completion_price_per_1k, prompt_format=args.prompt_format, few_shot_k=args.few_shot_k)
    
    # Gene pool initialization: dataset-specific valid alternating structures
    if dataset_string == 'bookcrossing':
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
    else:  # amazons
        AMAZON_UIU = [['U', 'I', 'U'], [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]]
        AMAZON_UIUI = [['U', 'I', 'U', 'I'], [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ]]
        AMAZON_UIUI_CYCLE = [['U', 'I', 'U', 'I'], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ]]
        AMAZON_UIUIU = [['U', 'I', 'U', 'I', 'U'], [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0]
        ]]
        AMAZON_UIUIUI = [['U', 'I', 'U', 'I', 'U', 'I'], [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0]
        ]]
        gene_pools = [AMAZON_UIU, AMAZON_UIUI, AMAZON_UIUI_CYCLE, AMAZON_UIUIU, AMAZON_UIUIUI]

    ##################################################################################################################
    # Optimization

    if(args.test_known_metas):
        print('***********Testing pre-identified meta-structures.*****************')

    performance_dict = {'best_val': {}, 'correspond_test': {}}
    gene_pools_performance_dict = {'best_val': {}, 'correspond_test': {}}
    full_logic_perf_dict = dict()
    best_performance = 0
    best_test = None
    best_gene_pools = gene_pools.copy()
    best_generation = None
    best_individual = None
    seen_structures = set()
    num_seen_structures = len(seen_structures)
    start0 = time.time()
    usage_start = get_llm_usage_counters()
    for gen in range(args.num_generations):
        gen_start = time.time()
        usage_before = get_llm_usage_counters()
        gene_pools_history_dict[gen] = gene_pools
        population_performance = []
        population_final_test = []
        new_gene_pools = []
        if(len(seen_structures)>num_seen_structures):
            num_seen_structures = len(seen_structures)
            print(f'Explored new meta-structure (s). Total exploration: {num_seen_structures}')

        eval_time_accum = 0.0
        for i in range(len(gene_pools)):
            gene = gene_pools[i]
            logic = structure2logic(gene, dataset_string)
            
            # Evaluation
            if(logic in performance_dict['best_val']):
                print('Evaluated.')
                performance = performance_dict['best_val'][logic]
                final_test = performance_dict['correspond_test'][logic]
            else:
                t0 = time.time()
                performance, final_test = evaluate([gene], dataset_string)
                eval_time_accum += (time.time() - t0)
                if(performance>best_performance):
                    best_performance = performance
                    best_test = final_test
                    best_gene = gene.copy()
                    best_logic = structure2logic(best_gene, dataset_string)
                    best_generation = gen
                    best_individual = i
                performance_dict['best_val'][logic] = performance
                performance_dict['correspond_test'][logic] = final_test
            print(f'Generation {gen}, Individual {i}, current performance: {performance}, best performance: {best_performance}, correspond. test: {best_test}, at Gen {best_generation} Individual {best_individual} \n')
            population_performance.append(performance)
            population_final_test.append(final_test)
            gene_pools_performance_dict['best_val'][gen] = population_performance.copy()
            gene_pools_performance_dict['correspond_test'][gen] = population_final_test.copy()

            full_logic_perf_dict[logic] = np.round(performance,6)

            most_recent_performances_prompt = ''
            best_performance_prompt = ''

        # Save performance
        if(not args.test_known_metas):
            with open(os.path.join(logdir, 'performance_dict.pkl'), 'wb') as f:
                pickle.dump(performance_dict, f)
            with open(os.path.join(logdir, 'gene_pools_history_dict.pkl'), 'wb') as f:
                pickle.dump(gene_pools_history_dict, f)    
            with open(os.path.join(logdir, 'gene_pools_performance_dict.pkl'), 'wb') as f:
                pickle.dump(gene_pools_performance_dict, f)    
        
        # Eliminate and reproduce
        population_performance = np.array(population_performance)
        gene_pools = eliminate_and_reproduce(gene_pools, population_performance)

        # Improve meta-structures
        llm_start = time.time()
        for gene in gene_pools:
            new_gene, seen_structures, _  = llm4meta.modify_metas(gen, [gene], seen_structures, most_recent_performances_prompt, best_performance_prompt, full_logic_perf_dict)
            new_gene_pools.append(new_gene)
        gene_pools = new_gene_pools.copy()
        llm_time = time.time() - llm_start
        gen_end = time.time()
        usage_after = get_llm_usage_counters()
        usage_delta = {
            "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
            "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
            "call_count": usage_after["call_count"] - usage_before["call_count"]
        }
        try:
            out_line = {
                "generation": gen,
                "eval_time_sec": eval_time_accum,
                "llm_time_sec": llm_time,
                "gen_total_sec": gen_end - gen_start,
                "llm_prompt_tokens": usage_delta["prompt_tokens"],
                "llm_completion_tokens": usage_delta["completion_tokens"],
                "llm_call_count": usage_delta["call_count"]
            }
            with open(os.path.join(logdir, 'cost_stats.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(out_line) + "\n")
        except Exception:
            pass
    
    total_time = time.time() - start0
    print(f'All {args.num_generations} generated. Used time: ', total_time)
    print(f'best_gene_pools: {best_gene_pools}, best val: {best_performance}, corrspond.test: {best_test}, at Generation {best_generation} Individual {best_individual}')
    try:
        usage_final = get_llm_usage_counters()
        summary = {
            "dataset": args.dataset,
            "dataset_seed": args.dataset_seed,
            "num_generations": args.num_generations,
            "population_size": POPULATION_SIZE,
            "epochs": args.epochs,
            "patience": args.patience,
            "llm_temperature": args.llm_temperature,
            "llm_top_p": args.llm_top_p,
            "llm_max_tokens": args.llm_max_tokens,
            "total_time_sec": total_time,
            "total_prompt_tokens": usage_final["prompt_tokens"],
            "total_completion_tokens": usage_final["completion_tokens"],
            "total_llm_calls": usage_final["call_count"]
        }
        with open(os.path.join(logdir, f'seed_summary_pf_{args.prompt_format}_fs_{args.few_shot_k}_seed_{args.seed}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(summary))
    except Exception:
        pass
    try:
        metrics = getattr(llm4meta, "get_prompt_metrics", None)
        if callable(metrics):
            m = llm4meta.get_prompt_metrics()
            with open(os.path.join(logdir, 'prompt_metrics.json'), 'w', encoding='utf-8') as f:
                f.write(json.dumps(m))
    except Exception:
        pass

    
