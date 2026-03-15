import torch
import torch.nn as nn
import torch.nn.functional as F

class UserPreferenceAggregator(nn.Module):
    def __init__(self, in_dim, query_dim):
        super().__init__()
        self.query_projection = nn.Linear(query_dim, in_dim)
        self.key_projection = nn.Linear(in_dim, in_dim)
        self.value_projection = nn.Linear(in_dim, in_dim)

    def forward(self, user_queries, item_features, user_history_map, all_item_indices):
        num_users = user_queries.size(0)
        device = user_queries.device

        # Project user queries
        projected_queries = self.query_projection(user_queries)

        # Prepare batch processing for users
        batch_user_indices = []
        batch_item_indices = []

        for user_idx in range(num_users):
            history = user_history_map.get(user_idx, [])
            if not history:
                # If no history, use a dummy item index that will be masked later
                history = [all_item_indices[0]] 

            batch_user_indices.extend([user_idx] * len(history))
            batch_item_indices.extend(history)

        # Convert to tensors
        batch_user_indices = torch.tensor(batch_user_indices, dtype=torch.long, device=device)
        batch_item_indices = torch.tensor(batch_item_indices, dtype=torch.long, device=device)
        
        # Fetch corresponding item features and project them
        keys = self.key_projection(item_features[batch_item_indices])
        values = self.value_projection(item_features[batch_item_indices])
        
        # Expand queries to match the batch size
        queries_expanded = projected_queries[batch_user_indices]

        # Calculate attention scores
        attn_scores = (queries_expanded * keys).sum(dim=1)

        # Create a mask for users with no history to avoid softmax over empty sets
        no_history_mask = torch.zeros(num_users, dtype=torch.bool, device=device)
        for i in range(num_users):
            if not user_history_map.get(i):
                no_history_mask[i] = True
        
        # Scatter max for stable softmax and compute softmax
        max_scores = torch.zeros(num_users, device=device).scatter_add_(0, batch_user_indices, attn_scores)
        max_scores_expanded = max_scores[batch_user_indices]
        exp_scores = torch.exp(attn_scores - max_scores_expanded)
        sum_exp_scores = torch.zeros(num_users, device=device).scatter_add_(0, batch_user_indices, exp_scores)
        sum_exp_scores_expanded = sum_exp_scores[batch_user_indices]
        
        attn_weights = exp_scores / (sum_exp_scores_expanded + 1e-12)

        # Apply attention weights to values
        weighted_values = values * attn_weights.unsqueeze(1)

        # Aggregate weighted values to form user profiles
        user_profiles = torch.zeros_like(projected_queries).scatter_add_(0, batch_user_indices.unsqueeze(1).expand(-1, values.size(1)), weighted_values)
        
        # For users with no history, use the projected query as their profile
        if no_history_mask.any():
            user_profiles[no_history_mask] = projected_queries[no_history_mask]
            
        return user_profiles
