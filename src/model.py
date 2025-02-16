import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Specialist Feed-Forward Network"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class DeepSeekMoE(nn.Module):
    """Camada MoE com balanceamento de carga auxiliar-free"""
    def __init__(self, num_experts, top_k, dim, hidden_dim):
        super().__init__()
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_experts)])  # noqa
        self.gate = nn.Linear(dim, num_experts)
        self.top_k = top_k
        self.bias = nn.Parameter(torch.zeros(num_experts))  # Termos de viés para balanceamento # noqa
        self.expert_usage = torch.zeros(num_experts)  # Track usage

    def update_balance(self, y=0.001):
        # Atualiza viéses baseado no uso dos experts
        mean_usage = self.expert_usage.mean()
        self.bias.data += y * (mean_usage - self.expert_usage)
        self.expert_usage.zero_()

    def forward(self, x):
        # x shape: [batch_size, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # flatten batch and sequence dims

        scores = self.gate(x_flat) + self.bias  # [batch*seq, num_experts]
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)

        # Registra uso dos experts
        for idx in top_indices.unique():
            self.expert_usage[idx] += (top_indices == idx).sum().item()

        gates = F.softmax(top_scores, dim=-1)  # [batch*seq, top_k]

        # Process all tokens in parallel
        outputs = torch.zeros_like(x_flat)  # [batch*seq, dim]
        for k in range(self.top_k):
            expert_indices = top_indices[:, k]  # [batch*seq]
            # Process each expert's assigned tokens
            for expert_idx in range(len(self.experts)):
                expert_mask = (expert_indices == expert_idx)
                if expert_mask.any():
                    tokens_for_expert = x_flat[expert_mask]  # Select tokens for this expert  # noqa
                    expert_output = self.experts[expert_idx](tokens_for_expert)
                    outputs[expert_mask] += gates[expert_mask, k].unsqueeze(-1) * expert_output  # noqa

        # Restore batch and sequence dimensions
        return outputs.view(batch_size, seq_len, dim)
