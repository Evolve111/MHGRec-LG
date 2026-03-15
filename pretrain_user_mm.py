import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from user_profile_aggregator import UserPreferenceAggregator
from utils import l2_normalize_rows
import pickle
import scipy.sparse as sp
from preprocess_recommendation import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor

def build_item_features(mm_dir, mm_item_type, mm_fusion, mm_alpha, mm_norm, num_items, device):
    text_path = os.path.join(mm_dir, 'embed_text.npy')
    img_path = os.path.join(mm_dir, 'embed_image.npy')
    text_emb = np.load(text_path) if os.path.isfile(text_path) else None
    img_emb = np.load(img_path) if os.path.isfile(img_path) else None
    if mm_norm:
        if text_emb is not None:
            text_emb = l2_normalize_rows(text_emb)
        if img_emb is not None:
            img_emb = l2_normalize_rows(img_emb)
    if text_emb is None and img_emb is None:
        raise ValueError("No multimodal embeddings found in mm_dir; cannot pretrain user profiles without item features.")
    if img_emb is not None and text_emb is not None:
        if mm_fusion == 'sum':
            fused = img_emb + text_emb
        elif mm_fusion == 'avg':
            fused = (img_emb + text_emb) / 2.0
        elif mm_fusion == 'weighted':
            fused = mm_alpha * img_emb + (1.0 - mm_alpha) * text_emb
        elif mm_fusion == 'concat':
            fused = np.concatenate([img_emb, text_emb], axis=1)
        elif mm_fusion == 'image':
            fused = img_emb
        elif mm_fusion == 'text':
            fused = text_emb
        else:
            fused = (img_emb + text_emb) / 2.0
    else:
        fused = text_emb if text_emb is not None else img_emb
    fused_item_feats = torch.from_numpy(fused).float().to(device)
    return fused_item_feats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Bookcrossing')
    parser.add_argument('--mm_dir', type=str, required=True, help='path to multimodal dataset folder')
    parser.add_argument('--mm_item_type', type=int, default=1)
    parser.add_argument('--mm_fusion', type=str, default='avg')
    parser.add_argument('--mm_alpha', type=float, default=0.5)
    parser.add_argument('--mm_norm', action='store_true', default=True)
    parser.add_argument('--user_type', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_seed', type=int, default=2)
    parser.add_argument('--out_path', type=str, default=None, help='path to save precomputed user profiles (.npy)')
    parser.add_argument('--user_mm_norm', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    prefix = os.path.join('data_recommendation', args.dataset)
    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    node_types_t = torch.from_numpy(node_types).to(device)
    num_node_types = node_types.max() + 1
    type_counts = [int((node_types_t == k).sum().item()) for k in range(num_node_types)]
    # load splits
    pos = np.load(os.path.join(prefix, f"pos_pairs_offset_larger_than_2_{args.dataset_seed}.npz"))
    pos_train = pos['train']
    neg = np.load(os.path.join(prefix, f"neg_pairs_offset_for_pos_larger_than_2_{args.dataset_seed}.npz"))
    neg_train = neg['train']

    # build item features
    fused_item_feats = build_item_features(args.mm_dir, args.mm_item_type, args.mm_fusion, args.mm_alpha, args.mm_norm, type_counts[args.mm_item_type], device)
    item_feature_dim = fused_item_feats.size(1)

    # build local index maps
    user_global_indices = torch.nonzero(node_types_t == args.user_type, as_tuple=False).squeeze(1).cpu().numpy().tolist()
    item_global_indices = torch.nonzero(node_types_t == args.mm_item_type, as_tuple=False).squeeze(1).cpu().numpy().tolist()
    user_global_to_local = {g: i for i, g in enumerate(user_global_indices)}
    item_global_to_local = {g: i for i, g in enumerate(item_global_indices)}

    # construct user history in local item index
    user_history_local = {}
    for pair in pos_train:
        u_g = int(pair[0]); i_g = int(pair[1])
        if (u_g not in user_global_to_local) or (i_g not in item_global_to_local):
            continue
        u_l = user_global_to_local[u_g]
        i_l = item_global_to_local[i_g]
        user_history_local.setdefault(u_l, []).append(i_l)

    # parameters: user queries and aggregator
    query_dim = item_feature_dim
    user_query_embedding = nn.Embedding(len(user_global_indices), query_dim).to(device)
    aggregator = UserPreferenceAggregator(in_dim=item_feature_dim, query_dim=query_dim).to(device)
    optimizer = torch.optim.Adam(list(aggregator.parameters()) + list(user_query_embedding.parameters()), lr=args.lr, weight_decay=args.wd)

    # train loop: logistic loss with dot(user_profile, item_feat)
    for epoch in range(args.epochs):
        aggregator.train()
        user_queries = user_query_embedding(torch.arange(len(user_global_indices), device=device))
        all_item_local_indices = torch.arange(len(item_global_indices), device=device)
        user_mm_profiles = aggregator(user_queries, fused_item_feats, user_history_local, all_item_local_indices)
        # normalize optionally
        if args.user_mm_norm:
            norms = torch.norm(user_mm_profiles, dim=1, keepdim=True) + 1e-12
            user_mm_profiles = user_mm_profiles / norms
        # positive scores
        pos_users_local = torch.tensor([user_global_to_local[int(u)] for u, _i in pos_train], dtype=torch.long, device=device)
        pos_items_local = torch.tensor([item_global_to_local[int(i)] for _u, i in pos_train], dtype=torch.long, device=device)
        pos_scores = (user_mm_profiles[pos_users_local] * fused_item_feats[pos_items_local]).sum(dim=1)
        # negative scores
        neg_users_local = torch.tensor([user_global_to_local[int(u)] for u, _i in neg_train], dtype=torch.long, device=device)
        neg_items_local = torch.tensor([item_global_to_local[int(i)] for _u, i in neg_train], dtype=torch.long, device=device)
        neg_scores = (user_mm_profiles[neg_users_local] * fused_item_feats[neg_items_local]).sum(dim=1)
        loss = - torch.mean(F.logsigmoid(pos_scores) + F.logsigmoid(-neg_scores))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info(f"Epoch {epoch+1}/{args.epochs} | Pretrain loss {loss.item():.4f}")

    # final profiles
    aggregator.eval()
    with torch.no_grad():
        user_queries = user_query_embedding(torch.arange(len(user_global_indices), device=device))
        all_item_local_indices = torch.arange(len(item_global_indices), device=device)
        user_mm_profiles = aggregator(user_queries, fused_item_feats, user_history_local, all_item_local_indices)
        if args.user_mm_norm:
            norms = torch.norm(user_mm_profiles, dim=1, keepdim=True) + 1e-12
            user_mm_profiles = user_mm_profiles / norms
    out_np = user_mm_profiles.detach().cpu().numpy()

    # save
    if args.out_path is None:
        out_path = os.path.join(prefix, f"user_mm_profiles_seed{args.seed}.npy")
    else:
        out_path = args.out_path
    np.save(out_path, out_np)
    print(f"Saved precomputed user profiles to: {out_path} (shape={out_np.shape})")

if __name__ == '__main__':
    main()
