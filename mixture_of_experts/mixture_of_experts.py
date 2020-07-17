import torch
from torch import nn
import torch.nn.functional as F

# constants

MIN_EXPERT_CAPACITY = 4

# helper functions

def default(val, default_val):
    return val if val is not None else default_val

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t):
    return F.pad(t, (0, 0, 1, 0)).cumsum(dim=-2)[..., :-1, :]

# classes

class Experts(nn.Module):
    def __init__(self, dim, num_experts = 16, hidden_dim = None, activation = nn.ReLU):
        super().__init__()
        hidden_dim = default(hidden_dim, dim * 4)

        self.w1 = nn.Parameter(torch.randn(num_experts, dim, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden_dim, dim))
        self.act = activation(inplace = True)

    def forward(self, x):
        hidden = torch.einsum('end,edh->enh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('enh,ehd->end', hidden, self.w2)
        return out

class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Linear(dim, num_gates, bias = False)

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance = None):
        b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        raw_gates = self.w_gating(x).softmax(dim=-1)

        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask
            gate_1 *= equals_one_mask
            density_1_proxy *= equals_one_mask
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask
            del greater_zero_mask

        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        density_1 = mask_1.mean(dim=-2)
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, 1e-9))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        position_in_expert_1 = cumsum_exclusive(mask_1) * mask_1
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        mask_1_flat = mask_1.sum(dim=-1)

        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat

        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * F.one_hot(position_in_expert_1.long())[..., None, :expert_capacity] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * F.one_hot(position_in_expert_2.long())[..., None, :expert_capacity]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss

class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        loss_coef = 1e-2):
        super().__init__()

        self.num_experts = num_experts
        self.gate = Top2Gating(dim, num_gates = num_experts)
        self.experts = Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation)
        self.loss_coef = loss_coef

    def forward(self, inputs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef
