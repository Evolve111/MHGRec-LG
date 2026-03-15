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
from model_recommendation import Model
from preprocess_recommendation import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor
from utils import *

import networkx as nx
from utils import convert_to_networkx_graph
import pdb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='Bookcrossing')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--dropout', type=float, default=0.2) 
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--non_symmetric', default=False, action='store_true')
parser.add_argument('--attn_dim', type=int, default=128)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--num_structures', type=int, default=1)
parser.add_argument('--loss_margin', type=float, default=0.3)
parser.add_argument('--dataset_seed', type=int, default=2)
parser.add_argument('--population_size', type=int, default=5)
parser.add_argument('--mm_dir', type=str, default='', help='path to multimodal dataset folder (e.g., 多模态数据集/bookcrossing-vit_bert)')
parser.add_argument('--mm_item_type', type=int, default=1, help='node type id for items to use multimodal embeddings')
parser.add_argument('--mm_fusion', type=str, default='avg', help='fusion method: sum|avg|weighted|concat|text|image')
parser.add_argument('--mm_alpha', type=float, default=0.5, help='alpha for weighted fusion (alpha*image + (1-alpha)*text)')
parser.add_argument('--mm_norm', default=True, action='store_true')
parser.add_argument('--topk', type=int, default=10, help='Top-K for ranking metrics')
parser.add_argument('--eval_neg_per_pos', type=int, default=99, help='Number of negatives per positive for user-level eval')

# Explicit user multimodal preferences (optional)
parser.add_argument('--use_user_mm', action='store_true', default=False, help='Use explicit user multimodal preference profiles')
parser.add_argument('--user_type', type=int, default=0, help='node type id for users')
parser.add_argument('--user_mm_method', type=str, default='attn', choices=['attn','avg'], help='aggregation method for user profiles')
parser.add_argument('--user_query_dim', type=int, default=None, help='query dim for attention aggregator (defaults to item feature dim)')
parser.add_argument('--user_mm_norm', action='store_true', default=False, help='L2-normalize user profiles after aggregation')
parser.add_argument('--user_mm_precomputed', type=str, default='', help='path to precomputed user profiles (.npy) in local user order')
parser.add_argument('--eval_scenarios', type=str, default='general', choices=['general','sparse','cold','all'], help='evaluation group: general|sparse(<=5)|cold(<=2)|all')
parser.add_argument('--eval_mode', type=str, default='full_items', choices=['sampled','multi_pos','full_items'], help='evaluation mode: sampled(1正+N负)、multi_pos(多正样候选)、full_items(全物品排序)')
parser.add_argument('--delta', type=float, default=0.8, help='相似度阈值（与搜索阶段保持一致）')
parser.add_argument('--patience', type=int, default=30, help='早停阈值')
args = parser.parse_args()
# 允许 Bookcrossing / Amazons / Movielens 三个数据集
assert args.dataset.lower() in ('bookcrossing','amazons','movielens'), f"Unsupported dataset: {args.dataset}"

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)

logdir = os.path.join("log/eval", args.dataset)
if not os.path.exists(logdir):
    os.makedirs(logdir)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(logdir, prefix + ".txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

## deprecated: single flag used only by eval model; training model ignores it
## keep args.num_structures for structure selection but do not pass "single" into Model

def main():
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    connection_dict_s = {}
    connection_dict_t = {}

    # Align with generator's output directory naming (includes `_changeinit`) and δ
    dir = f'./log_recommendation/train_threshold_2_delta_{args.delta}_datasetseed_{args.dataset_seed}_changeinit/{args.dataset}' 
    gene_pools_history_dict_path = os.path.join(dir, 'gene_pools_history_dict.pkl')
    gene_pools_performance_dict_path = os.path.join(dir, 'gene_pools_performance_dict.pkl')
    with open(gene_pools_history_dict_path, 'rb') as f:
        gene_pools_history_dict = pickle.load(f)
    with open(gene_pools_performance_dict_path, 'rb') as f:
        gene_pools_performance_dict = pickle.load(f)
    print(gene_pools_history_dict.keys())
    gene_pools_best_val_dict = gene_pools_performance_dict['best_val']
    gene_pools_correspond_test_dict = gene_pools_performance_dict['correspond_test']
    best_val_array = np.zeros((len(gene_pools_history_dict.keys()), args.population_size))
    for generation,perfs in gene_pools_best_val_dict.items():
        best_val_array[generation] = perfs
    # Flatten the array to find indices of the largest elements
    flat_indices = np.argsort(best_val_array.flatten())[-args.num_structures:][::-1]
    # Convert flat indices to (row, col) indices
    row_indices, col_indices = np.unravel_index(flat_indices, best_val_array.shape)
    gene_comb = [gene_pools_history_dict[row_indices[i]][col_indices[i]] for i in range(args.num_structures)]
    individual_best_val = [gene_pools_best_val_dict[row_indices[i]][col_indices[i]] for i in range(args.num_structures)]
    individual_correspond_test = [gene_pools_correspond_test_dict[row_indices[i]][col_indices[i]] for i in range(args.num_structures)]
    
    print('Structures: ', gene_comb)
    print('Individual Best val: ', individual_best_val)
    print('Individual Correspond.test: ', individual_correspond_test)
    
    # Set a meta
    test_structure_list_sym = [gene_comb[i] for i in range(args.num_structures)]
    print('len(test_structure_list_sym): ', len(test_structure_list_sym))
    test_structure_list_source = [gene_comb[int(2*i)] for i in range(int(args.num_structures/2))] 
    test_structure_list_target = [gene_comb[int(2*i+1)] for i in range(int(args.num_structures/2))] 
    # removed debugging breakpoint

    archs = {args.dataset: {'source': ([],[]), 'target': ([],[])}} # Initialization
    edge_type_lookup_dict = temp_edge_type_lookup(args.dataset.lower())
    if(not args.non_symmetric):
        for test_structure in test_structure_list_sym:
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_target = construct_arch(G, state_list, edge_type_lookup_dict) 
            state_list = get_state_list(G, 0)
            arch_source = construct_arch(G, state_list, edge_type_lookup_dict) 
            archs[args.dataset]['source'][0].append(arch_source[0])
            archs[args.dataset]['source'][1].append(arch_source[1])
            archs[args.dataset]['target'][0].append(arch_target[0])
            archs[args.dataset]['target'][1].append(arch_target[1])
    else:
        for test_structure in test_structure_list_source:
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_source = construct_arch(G, state_list, edge_type_lookup_dict) 
            archs[args.dataset]['source'][0].append(arch_source[0])
            archs[args.dataset]['source'][1].append(arch_source[1])
        for test_structure in test_structure_list_target:   
            G = convert_to_networkx_graph({'nodes': test_structure[0], 'edges': test_structure[1]})
            state_list = get_state_list(G, 1)
            arch_target = construct_arch(G, state_list, edge_type_lookup_dict) 
            archs[args.dataset]['target'][0].append(arch_target[0])
            archs[args.dataset]['target'][1].append(arch_target[1])

    steps_s = [len(meta) for meta in archs[args.dataset]["source"][0]] 
    steps_t = [len(meta) for meta in archs[args.dataset]["target"][0]] 

    datadir = 'data_recommendation' 
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
    zero_indices = torch.empty((2, 0), dtype=torch.long)
    zero_values = torch.empty((0,), dtype=torch.float32)
    adjs_pt.append(torch.sparse_coo_tensor(zero_indices, zero_values, adjs_offset['1'].shape, dtype=torch.float32, device=device))
    print("Loading {} adjs...".format(len(adjs_pt)))

    #* load labels
    pos = np.load(os.path.join(prefix, f"pos_pairs_offset_larger_than_2_{args.dataset_seed}.npz"))
    pos_train = pos['train']
    pos_val = pos['val']
    pos_test = pos['test']

    neg = np.load(os.path.join(prefix, f"neg_pairs_offset_for_pos_larger_than_2_{args.dataset_seed}.npz"))
    neg_train = neg['train']
    neg_val = neg['val']
    neg_test = neg['test']

    # Optional global negative pool for user-level sampling
    neg_global_path = os.path.join(prefix, 'neg_ratings_offset_smaller_than_3.npy')
    global neg_global
    neg_global = None
    if os.path.isfile(neg_global_path):
        try:
            neg_global = np.load(neg_global_path)
        except Exception:
            neg_global = None

    #* build heterogeneous node features: users one-hot; items multimodal embeddings if provided
    type_counts = [int((node_types == k).sum().item()) for k in range(num_node_types)]
    in_dims = type_counts.copy()
    node_feats = []

    fused_item_feats = None
    item_feature_dim = None
    if args.mm_dir and os.path.isdir(args.mm_dir):
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
                text_emb = l2_normalize_rows(text_emb)
            if img_emb is not None:
                img_emb = l2_normalize_rows(img_emb)

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
            item_feature_dim = fused_item_feats.size(1)
            in_dims[args.mm_item_type] = item_feature_dim

    # Build user-item history map from training positives (global indices)
    # and convert to local indices per type for stable alignment
    if args.user_mm_precomputed and os.path.isfile(args.user_mm_precomputed):
        node_types_tensor = torch.from_numpy(node_types).to(device) if not torch.is_tensor(node_types) else node_types
        user_global_indices = torch.nonzero(node_types_tensor == args.user_type, as_tuple=False).squeeze(1).cpu().numpy().tolist()
        # Load precomputed profiles in local user order
        precomp = np.load(args.user_mm_precomputed)
        assert precomp.shape[0] == len(user_global_indices), f"Precomputed profiles rows ({precomp.shape[0]}) != num_users ({len(user_global_indices)})"
        user_mm_profiles = torch.from_numpy(precomp).float().to(device)
        if args.user_mm_norm:
            norms = torch.norm(user_mm_profiles, dim=1, keepdim=True) + 1e-12
            user_mm_profiles = user_mm_profiles / norms
        in_dims[args.user_type] = user_mm_profiles.size(1)
    elif args.use_user_mm and fused_item_feats is not None:
        from user_profile_aggregator import UserPreferenceAggregator
        import torch.nn as nn

        node_types_tensor = torch.from_numpy(node_types).to(device) if not torch.is_tensor(node_types) else node_types
        # Get global indices for user and item types
        user_global_indices = torch.nonzero(node_types_tensor == args.user_type, as_tuple=False).squeeze(1).cpu().numpy().tolist()
        item_global_indices = torch.nonzero(node_types_tensor == args.mm_item_type, as_tuple=False).squeeze(1).cpu().numpy().tolist()
        # Build global->local index maps
        user_global_to_local = {g: i for i, g in enumerate(user_global_indices)}
        item_global_to_local = {g: i for i, g in enumerate(item_global_indices)}

        # Construct user history map with local item indices
        user_history_local = {}
        for pair in pos_train:
            u_g = int(pair[0])
            i_g = int(pair[1])
            # Skip if not user->item by type
            if (u_g not in user_global_to_local) or (i_g not in item_global_to_local):
                continue
            u_l = user_global_to_local[u_g]
            i_l = item_global_to_local[i_g]
            user_history_local.setdefault(u_l, []).append(i_l)

        # Prepare user queries embedding
        num_users = len(user_global_indices)
        query_dim = args.user_query_dim if args.user_query_dim is not None else item_feature_dim
        user_query_embedding = nn.Embedding(num_users, query_dim).to(device)

        if args.user_mm_method == 'attn':
            aggregator = UserPreferenceAggregator(in_dim=item_feature_dim, query_dim=query_dim).to(device)
            # Compute user profiles via attention aggregator
            user_queries = user_query_embedding(torch.arange(num_users, device=device))
            all_item_local_indices = torch.arange(len(item_global_indices), device=device)
            user_mm_profiles = aggregator(user_queries, fused_item_feats, user_history_local, all_item_local_indices)
        else:
            # Simple average pooling over history
            user_mm_profiles = torch.zeros((num_users, item_feature_dim), device=device)
            for u_l, items_l in user_history_local.items():
                if len(items_l) == 0:
                    continue
                feats = fused_item_feats[torch.tensor(items_l, dtype=torch.long, device=device)]
                user_mm_profiles[u_l] = feats.mean(dim=0)
            # For users with no history, initialize from embedding projection
            if (len(user_history_local) < num_users):
                user_queries = user_query_embedding(torch.arange(num_users, device=device))
                proj = nn.Linear(query_dim, item_feature_dim).to(device)
                user_mm_profiles[user_mm_profiles.sum(dim=1) == 0] = proj(user_queries[user_mm_profiles.sum(dim=1) == 0])

        # Optional normalization
        if args.user_mm_norm:
            norms = torch.norm(user_mm_profiles, dim=1, keepdim=True) + 1e-12
            user_mm_profiles = user_mm_profiles / norms

        # Replace user one-hot with dense profiles and update in_dims
        in_dims[args.user_type] = item_feature_dim

    for k in range(num_node_types):
        if fused_item_feats is not None and k == args.mm_item_type:
            node_feats.append(fused_item_feats)
        elif k == args.user_type and (
            (args.use_user_mm and fused_item_feats is not None) or
            (args.user_mm_precomputed and os.path.isfile(args.user_mm_precomputed))
        ):
            node_feats.append(user_mm_profiles)
        else:
            count_k = type_counts[k]
            i = torch.stack((torch.arange(count_k, dtype=torch.long), torch.arange(count_k, dtype=torch.long)))
            v = torch.ones(count_k, dtype=torch.float32)
            node_feats.append(torch.sparse_coo_tensor(i, v, torch.Size([count_k, count_k]), dtype=torch.float32, device=device))

    assert(len(in_dims) == len(node_feats))   

    # Train-time Model does not accept 'single' or 'num_heads'
    model_s = Model(in_dims, args.n_hid, steps_s, dropout=args.dropout, attn_dim=args.attn_dim).to(device)
    model_t = Model(in_dims, args.n_hid, steps_t, dropout=args.dropout, attn_dim=args.attn_dim).to(device)

    # Include aggregator parameters if used
    opt_params = list(model_s.parameters()) + list(model_t.parameters())
    if (args.use_user_mm and fused_item_feats is not None) and not (args.user_mm_precomputed and os.path.isfile(args.user_mm_precomputed)):
        # Add aggregator and embedding parameters where applicable
        if args.user_mm_method == 'attn':
            opt_params += list(aggregator.parameters())
        opt_params += list(user_query_embedding.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=args.lr, weight_decay=args.wd)

    best_val = None
    final = None
    anchor = None
    no_improve = 0
    for epoch in range(args.epochs):
        # Recompute user multimodal profiles with current aggregator params each epoch
        if args.use_user_mm and fused_item_feats is not None:
            if args.user_mm_method == 'attn':
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
                    proj = torch.nn.Linear(user_queries.size(1), item_feature_dim).to(device)
                    user_mm_profiles[user_mm_profiles.sum(dim=1) == 0] = proj(user_queries[user_mm_profiles.sum(dim=1) == 0])
            if args.user_mm_norm:
                norms = torch.norm(user_mm_profiles, dim=1, keepdim=True) + 1e-12
                user_mm_profiles = user_mm_profiles / norms
            node_feats[args.user_type] = user_mm_profiles

        train_loss = train(archs, node_feats, node_types, adjs_pt, pos_train, neg_train, model_s, model_t, connection_dict_s, connection_dict_t, optimizer)
        val_loss, auc_val, auc_test, wrong_predictions = infer(archs, node_feats, node_types, adjs_pt, pos_train, pos_val, neg_val, pos_test, neg_test, model_s, model_t, connection_dict_s, connection_dict_t)
        logging.info("Epoch {}; Train err {}; Val err {}; Val auc {}".format(epoch + 1, train_loss, val_loss, auc_val))
        if best_val is None or auc_val > best_val:
            best_val = auc_val
            final = auc_test
            anchor = epoch + 1
            final_wrong_predictions = wrong_predictions.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break
    logging.info("Best val auc {} at epoch {}; Test auc {}".format(best_val, anchor, final))

    try:
        with torch.no_grad():
            out_s_final = model_s(node_feats, node_types, adjs_pt, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1], connection_dict_s)
            out_t_final = model_t(node_feats, node_types, adjs_pt, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1], connection_dict_t)
            out_s_final = torch.nan_to_num(out_s_final, nan=0.0, posinf=0.0, neginf=0.0)
            out_t_final = torch.nan_to_num(out_t_final, nan=0.0, posinf=0.0, neginf=0.0)
        node_types_arr = node_types.cpu().numpy()
        item_type_id = args.mm_item_type
        scenarios_to_run = ['general','sparse','cold'] if args.eval_scenarios == 'all' else [args.eval_scenarios]
        def filter_by_scenario_final(pos_pairs, neg_pairs, scenario: str, train_pairs):
            if scenario not in ('sparse','cold'):
                return pos_pairs, neg_pairs
            from collections import defaultdict
            train_counts = defaultdict(int)
            for u, _i in train_pairs:
                train_counts[int(u)] += 1
            if scenario == 'sparse':
                selected_users = {u for u, c in train_counts.items() if c <= 5}
            else:
                selected_users = {u for u, c in train_counts.items() if c <= 2}
            pos_filtered = np.array([p for p in pos_test if int(p[0]) in selected_users], dtype=pos_test.dtype)
            neg_filtered = np.array([n for n in neg_test if int(n[0]) in selected_users], dtype=neg_test.dtype)
            return pos_filtered, neg_filtered
        # global neg pool
        neg_global_path = os.path.join(os.path.join('data_recommendation', args.dataset), 'neg_ratings_offset_smaller_than_3.npy')
        neg_global_arr = np.load(neg_global_path) if os.path.isfile(neg_global_path) else None
        for scen in scenarios_to_run:
            if scen == 'general':
                pos_use, neg_use = pos_test, neg_test
            else:
                pos_use, neg_use = filter_by_scenario_final(pos_test, neg_test, scen, pos_train)
            # metrics
            def compute_user_topk_metrics_final(pos_pairs, neg_pairs, out_s, out_t, topk: int, neg_per_pos: int, node_types_arr, item_type_id, neg_global_arr=None):
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
                    if len(negs_u) < args.eval_neg_per_pos:
                        extras = [i for i in item_indices if i != pos_item and i not in negs_u]
                        if len(extras) > 0:
                            random.shuffle(extras)
                            take = min(args.eval_neg_per_pos - len(negs_u), len(extras))
                            negs_u.extend(extras[:take])
                    if len(negs_u) > args.eval_neg_per_pos:
                        random.shuffle(negs_u)
                        negs_u = negs_u[:args.eval_neg_per_pos]
                    cand_items = [pos_item] + negs_u
                    scores = (out_s[u].unsqueeze(0) * out_t[cand_items]).sum(dim=-1)
                    topk_use = min(args.topk, len(cand_items))
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
            recall10, precision10, ndcg10, hr10 = compute_user_topk_metrics_final(
                pos_use, neg_use, out_s_final, out_t_final, 10, args.eval_neg_per_pos, node_types_arr, item_type_id,
                neg_global_arr=neg_global_arr
            )
            recall20, precision20, ndcg20, hr20 = compute_user_topk_metrics_final(
                pos_use, neg_use, out_s_final, out_t_final, 20, args.eval_neg_per_pos, node_types_arr, item_type_id,
                neg_global_arr=neg_global_arr
            )
            try:
                pos_prod_scen = torch.mul(out_s_final[pos_use[:, 0]], out_t_final[pos_use[:, 1]]).sum(dim=-1)
                neg_prod_scen = torch.mul(out_s_final[neg_use[:, 0]], out_t_final[neg_use[:, 1]]).sum(dim=-1)
                y_true_scen = np.zeros((pos_use.shape[0] + neg_use.shape[0]), dtype=np.long)
                y_true_scen[:pos_use.shape[0]] = 1
                y_pred_scen = np.concatenate((torch.sigmoid(pos_prod_scen).cpu().numpy(), torch.sigmoid(neg_prod_scen).cpu().numpy()))
                auc_scen = roc_auc_score(y_true_scen, y_pred_scen) if y_true_scen.sum() > 0 else 0.0
            except Exception:
                auc_scen = final if final is not None else 0.0
            msg_final = f"RESULT [group={scen}] AUC {auc_scen:.4f} | P@10 {precision10:.4f} | R@10 {recall10:.4f} | P@20 {precision20:.4f} | R@20 {recall20:.4f}"
            print(msg_final)
            try:
                file_out = os.path.join('log/eval', args.dataset, prefix + ".txt")
                with open(file_out, 'a', encoding='utf-8') as f:
                    f.write(msg_final + "\n")
                file_out2 = os.path.join(os.getcwd(), f"results_variantA_book_seed{args.seed}.txt")
                with open(file_out2, 'a', encoding='utf-8') as f2:
                    f2.write(msg_final + "\n")
            except Exception:
                pass
    except Exception:
        pass

def train(archs, node_feats, node_types, adjs, pos_train, neg_train, model_s, model_t, connection_dict_s, connection_dict_t, optimizer):

    model_s.train()
    model_t.train()
    optimizer.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1], connection_dict_s)
    out_t = model_t(node_feats, node_types, adjs, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1], connection_dict_t)
    out_s = torch.nan_to_num(out_s, nan=0.0, posinf=0.0, neginf=0.0)
    out_t = torch.nan_to_num(out_t, nan=0.0, posinf=0.0, neginf=0.0)
    pos_logits = torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)
    neg_logits = torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)
    pos_logits = torch.clamp(pos_logits, -50.0, 50.0)
    neg_logits = torch.clamp(neg_logits, -50.0, 50.0)
    loss = - torch.mean(F.logsigmoid(pos_logits) + F.logsigmoid(- neg_logits))
    loss.backward()
    optimizer.step()
    return loss.item()

def infer(archs, node_feats, node_types, adjs, pos_train, pos_val, neg_val, pos_test, neg_test, model_s, model_t, connection_dict_s, connection_dict_t):

    model_s.eval()
    model_t.eval()
    with torch.no_grad():
        out_s = model_s(node_feats, node_types, adjs, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1], connection_dict_s)
        out_t = model_t(node_feats, node_types, adjs, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1], connection_dict_t)
        out_s = torch.nan_to_num(out_s, nan=0.0, posinf=0.0, neginf=0.0)
        out_t = torch.nan_to_num(out_t, nan=0.0, posinf=0.0, neginf=0.0)
    
    #* validation performance
        pos_val_prod = torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)
    neg_val_prod = torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)
    loss = - torch.mean(F.logsigmoid(pos_val_prod) + F.logsigmoid(- neg_val_prod))
    y_true_val = np.zeros((pos_val.shape[0] + neg_val.shape[0]), dtype=np.long)
    y_true_val[:pos_val.shape[0]] = 1
    y_pred_val = np.concatenate((torch.sigmoid(pos_val_prod).cpu().numpy(), torch.sigmoid(neg_val_prod).cpu().numpy()))
    y_pred_val = np.nan_to_num(y_pred_val, nan=0.5, posinf=1.0, neginf=0.0)
    auc_val = roc_auc_score(y_true_val, y_pred_val)
    y_pred_val_binary = np.where(y_pred_val < 0.5, 0, 1)
    wrong_predictions = np.where(y_pred_val_binary != y_true_val)[0]

    #* test performance
    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)

    y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.long)
    y_true_test[:pos_test.shape[0]] = 1
    y_pred_test = np.concatenate((torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))
    y_pred_test = np.nan_to_num(y_pred_test, nan=0.5, posinf=1.0, neginf=0.0)
    auc_test = roc_auc_score(y_true_test, y_pred_test)

    # User-level Top-K metrics: for each user, 1 positive + N negatives
    def compute_user_topk_metrics(pos_pairs, neg_pairs, out_s, out_t, topk: int, neg_per_pos: int, node_types_arr, item_type_id, neg_global_arr=None):
        import random
        random.seed(args.seed)
        from collections import defaultdict

        # Build user -> positives list
        user_pos = defaultdict(list)
        for u, i in pos_pairs:
            user_pos[int(u)].append(int(i))

        # Build user -> negatives list from provided neg_pairs
        user_negs = defaultdict(list)
        for u, i in neg_pairs:
            user_negs[int(u)].append(int(i))

        # Item pool indices
        item_indices = np.where(node_types_arr == item_type_id)[0].tolist()

        users = list(user_pos.keys())
        hits = 0
        precision_sum = 0.0
        dcg_sum = 0.0
        idcg_sum = float(len(users))  # with one relevant, IDCG@k is 1 per user
        for u in users:
            # pick one positive item for this user
            pos_items = user_pos[u]
            if len(pos_items) == 0:
                continue
            pos_item = pos_items[0]

            # collect negatives for this user
            negs_u = user_negs[u][:]
            if neg_global_arr is not None:
                # add more negatives for this user from global pool
                # filter by user u
                mask = (neg_global_arr[:,0] == u)
                negs_u.extend([int(x) for x in neg_global_arr[mask][:,1].tolist()])

            # remove duplicates and positives
            negs_u = list({i for i in negs_u if i != pos_item})
            # if still insufficient, sample from item pool excluding pos
            if len(negs_u) < neg_per_pos:
                extras = [i for i in item_indices if i != pos_item and i not in negs_u]
                if len(extras) > 0:
                    random.shuffle(extras)
                    take = min(neg_per_pos - len(negs_u), len(extras))
                    negs_u.extend(extras[:take])

            # finally sample exactly neg_per_pos
            if len(negs_u) > neg_per_pos:
                random.shuffle(negs_u)
                negs_u = negs_u[:neg_per_pos]

            # candidate set: 1 positive + N negatives
            cand_items = [pos_item] + negs_u
            scores = (out_s[u].unsqueeze(0) * out_t[cand_items]).sum(dim=-1)
            # ranks: higher score better
            topk_use = min(topk, len(cand_items))
            topk_indices = torch.topk(scores, topk_use).indices.tolist()
            topk_items = [cand_items[idx] for idx in topk_indices]
            hit = 1 if pos_item in topk_items else 0
            hits += hit
            precision_sum += hit / float(topk_use)
            # NDCG with one relevant
            # rank starts at 1
            sorted_idx = torch.argsort(scores, descending=True).tolist()
            rank_pos = sorted_idx.index(cand_items.index(pos_item)) + 1
            dcg_sum += 1.0 / np.log2(rank_pos + 1)

        users_count = len(users)
        hr = hits / float(users_count) if users_count > 0 else 0.0
        recall = hr  # with one relevant per user
        precision = precision_sum / float(users_count) if users_count > 0 else 0.0
        ndcg = dcg_sum / float(idcg_sum) if users_count > 0 else 0.0
        return recall, precision, ndcg, hr

    # multi-positive metrics on candidate set: candidates = positives ∪ negatives per user
    def compute_topk_metrics_multi(pos_pairs, neg_pairs, out_s, out_t, k: int = 10):
        from collections import defaultdict
        user_pos = defaultdict(list)
        user_cand = defaultdict(set)
        for u, i in pos_pairs:
            uu = int(u); ii = int(i)
            user_pos[uu].append(ii)
            user_cand[uu].add(ii)
        for u, i in neg_pairs:
            uu = int(u); ii = int(i)
            user_cand[uu].add(ii)

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
            scores = (out_s[u].unsqueeze(0) * out_t[items]).sum(dim=-1)
            topk = min(k, len(items))
            topk_indices = torch.topk(scores, topk).indices.tolist()
            topk_items = [items[idx] for idx in topk_indices]
            user_hits = len(set(pos_items) & set(topk_items))
            hits_total += user_hits
            if user_hits > 0:
                users_hit += 1
            precision_total += user_hits / float(topk)
            sorted_indices = torch.argsort(scores, descending=True).tolist()
            item2rank = {items[idx]: (r + 1) for r, idx in enumerate(sorted_indices)}
            dcg = 0.0
            for item in pos_items:
                rank = item2rank.get(item, None)
                if rank is not None and rank <= k:
                    dcg += 1.0 / np.log2(rank + 1)
            p_k = min(len(pos_items), k)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(p_k))
            dcg_total += dcg
            idcg_total += idcg
            users_count += 1

        recall_at_k = hits_total / pos_total if pos_total > 0 else 0.0
        precision_at_k = precision_total / users_count if users_count > 0 else 0.0
        ndcg_at_k = (dcg_total / idcg_total) if idcg_total > 0 else 0.0
        hitrate_at_k = users_hit / users_count if users_count > 0 else 0.0
        return recall_at_k, precision_at_k, ndcg_at_k, hitrate_at_k

    # full items metrics: rank over all items of given type
    def compute_topk_metrics_full_items(pos_pairs, out_s, out_t, k: int, node_types_arr, item_type_id):
        from collections import defaultdict
        user_pos = defaultdict(list)
        for u, i in pos_pairs:
            user_pos[int(u)].append(int(i))
        item_mask = (node_types_arr == item_type_id)
        all_items = np.where(item_mask)[0].tolist()
        hits_total = 0
        precision_total = 0.0
        users_count = 0
        pos_total = 0
        for u, pos_items in user_pos.items():
            if len(pos_items) == 0:
                continue
            pos_total += len(pos_items)
            scores = (out_s[u].unsqueeze(0) * out_t[all_items]).sum(dim=-1)
            topk_use = min(k, len(all_items))
            topk_idx = torch.topk(scores, topk_use).indices.tolist()
            topk_items = [all_items[idx] for idx in topk_idx]
            hits = len(set(pos_items) & set(topk_items))
            hits_total += hits
            precision_total += hits / float(topk_use)
            users_count += 1
        recall_at_k = hits_total / pos_total if pos_total > 0 else 0.0
        precision_at_k = precision_total / users_count if users_count > 0 else 0.0
        return recall_at_k, precision_at_k

    # optional scenario filtering based on training interaction counts
    def filter_by_scenario(pos_pairs, neg_pairs, scenario: str, train_pairs):
        if scenario not in ('sparse','cold'):
            return pos_pairs, neg_pairs
        from collections import defaultdict
        train_counts = defaultdict(int)
        for u, _i in train_pairs:
            train_counts[int(u)] += 1
        if scenario == 'sparse':
            selected_users = {u for u, c in train_counts.items() if c <= 5}
        else:  # cold
            selected_users = {u for u, c in train_counts.items() if c <= 2}
        pos_filtered = np.array([p for p in pos_pairs if int(p[0]) in selected_users], dtype=pos_pairs.dtype)
        neg_filtered = np.array([n for n in neg_pairs if int(n[0]) in selected_users], dtype=neg_pairs.dtype)
        return pos_filtered, neg_filtered

    try:
        # node_types is a tensor; get numpy
        node_types_arr = node_types.cpu().numpy()
        item_type_id = args.mm_item_type
        scenarios_to_run = []
        if args.eval_scenarios == 'all':
            scenarios_to_run = ['general','sparse','cold']
        else:
            scenarios_to_run = [args.eval_scenarios]
        for scen in scenarios_to_run:
            if scen == 'general':
                pos_use, neg_use = pos_test, neg_test
            else:
                pos_use, neg_use = filter_by_scenario(pos_test, neg_test, scen, pos_train)
            if args.eval_mode == 'sampled':
                recall10, precision10, ndcg10, hr10 = compute_user_topk_metrics(
                    pos_use, neg_use, out_s, out_t, args.topk, args.eval_neg_per_pos, node_types_arr, item_type_id,
                    neg_global_arr=neg_global if 'neg_global' in globals() else None
                )
                recall20, precision20, ndcg20, hr20 = compute_user_topk_metrics(
                    pos_use, neg_use, out_s, out_t, 20, args.eval_neg_per_pos, node_types_arr, item_type_id,
                    neg_global_arr=neg_global if 'neg_global' in globals() else None
                )
            elif args.eval_mode == 'multi_pos':
                recall10, precision10, ndcg10, hr10 = compute_topk_metrics_multi(pos_use, neg_use, out_s, out_t, k=10)
                recall20, precision20, ndcg20, hr20 = compute_topk_metrics_multi(pos_use, neg_use, out_s, out_t, k=20)
            else:  # full_items
                r10, p10 = compute_topk_metrics_full_items(pos_use, out_s, out_t, 10, node_types_arr, item_type_id)
                r20, p20 = compute_topk_metrics_full_items(pos_use, out_s, out_t, 20, node_types_arr, item_type_id)
                recall10, precision10, ndcg10, hr10 = r10, p10, 0.0, 0.0
                recall20, precision20, ndcg20, hr20 = r20, p20, 0.0, 0.0
            # compute AUC on filtered subset
            try:
                pos_prod_scen = torch.mul(out_s[pos_use[:, 0]], out_t[pos_use[:, 1]]).sum(dim=-1)
                neg_prod_scen = torch.mul(out_s[neg_use[:, 0]], out_t[neg_use[:, 1]]).sum(dim=-1)
                y_true_scen = np.zeros((pos_use.shape[0] + neg_use.shape[0]), dtype=np.long)
                y_true_scen[:pos_use.shape[0]] = 1
                y_pred_scen = np.concatenate((torch.sigmoid(pos_prod_scen).cpu().numpy(), torch.sigmoid(neg_prod_scen).cpu().numpy()))
                auc_scen = roc_auc_score(y_true_scen, y_pred_scen) if y_true_scen.sum() > 0 else 0.0
            except Exception:
                auc_scen = auc_test
            msg = f"RESULT [group={scen}] AUC {auc_scen:.4f} | P@10 {precision10:.4f} | R@10 {recall10:.4f} | P@20 {precision20:.4f} | R@20 {recall20:.4f}"
            logging.info(msg)
            try:
                file_out = os.path.join('log/eval', args.dataset, prefix + ".txt")
                with open(file_out, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
                file_out2 = os.path.join(os.getcwd(), f"results_variantA_book_seed{args.seed}.txt")
                with open(file_out2, 'a', encoding='utf-8') as f2:
                    f2.write(msg + "\n")
            except Exception:
                pass
    except Exception as e:
        logging.warning(f"Top-{args.topk} metric computation failed: {e}")

    return loss.item(), auc_val, auc_test, wrong_predictions


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


if __name__ == '__main__':
    main()
