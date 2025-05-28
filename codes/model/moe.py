import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Callable

class GLU(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable[[Tensor], Tensor],
        mult_bias: bool = False,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate) * self.mult_bias

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


def exists(val):
    return val is not None


def default(val, default_val):
    return default_val if val is None else val


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        mult=4,
        glu=False,
        glu_mult_bias=False,
        swish=False,
        relu_squared=False,
        post_act_ln=False,
        dropout: float = 0.0,
        no_bias=False,
        zero_init_output=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(
                dim, inner_dim, activation, mult_bias=glu_mult_bias
            )
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=not no_bias), activation
            )

        if post_act_ln:
            self.ff = nn.Sequential(
                project_in,
                nn.LayerNorm(inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=not no_bias),
            )
        else:
            self.ff = nn.Sequential(
                project_in,
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=not no_bias),
            )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)


class SwitchGate(nn.Module):
    def __init__(
        self,
        dim,
        num_experts: int,
        top_k: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor):
        gate_scores = F.softmax(self.w_gate(x), dim=-1)
        capacity = int(self.capacity_factor * x.size(0))
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(gate_scores).scatter_(1, top_k_indices, 1)
        masked_gate_scores = gate_scores * mask
        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        gate_scores = (masked_gate_scores / denominators) * capacity

        return gate_scores


class SwitchMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float = 1.0,
        mult: int = 4
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult

        self.experts = nn.ModuleList(
            [
                FeedForward(dim, output_dim, mult)
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(
            dim,
            num_experts,
            top_k,
            capacity_factor,
        )

    def forward(self, x: Tensor, gate_input: Tensor):
        # (batch_size, seq_len, num_experts)
        gate_scores = self.gate(gate_input)
        expert_outputs = [expert(x) for expert in self.experts]

        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[torch.isnan(stacked_expert_outputs)] = 0

        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output