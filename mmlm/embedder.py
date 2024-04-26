import math
import torch
from torch import nn
import torch.nn.functional as F


class LinearAdapter(nn.Module):
    def __init__(self, input_size, output_size, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter = nn.Linear(input_size, output_size).to(device)

    def forward(self, x):
        return self.adapter(x)


class WeightedSumAdapter(nn.Module):
    def __init__(self, input_size, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.ones(input_size)).to(device)
        self.softmax = nn.Softmax(dim=0).to(device)

    def forward(self, x):
        return torch.sum(x * self.softmax(self.weights), dim=1)


class CNNDownSampleAdapter(nn.Module):
    def __init__(self, input_size, output_size, downrate=8, device="cuda"):
        super().__init__()
        downloop = int(math.log(downrate, 2))
        self.adapters = nn.Sequential(*[
            nn.Conv1d(in_channels=input_size if i == 0 else output_size,
                      out_channels=output_size,
                      kernel_size=2,
                      stride=2) for i in range(downloop)
        ]).to(device)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.adapters(x).transpose(1, 2)


class RQCodeEmbedAdapter(nn.Module):
    def __init__(self, code_size, embed_size=768, embed_scale=1.0, normalized=True):
        super().__init__()
        self.code_size = code_size
        learning_weight_init = torch.arange(code_size, 0, step=-1).float().view(1, 1, code_size, 1)
        self.weighted_sum = nn.Parameter(learning_weight_init)
        self.embed_scale = embed_scale
        self.normalized = normalized
        self.embed_size = embed_size
        self.layer_norm = nn.LayerNorm(embed_size)

    def __call__(self, input_ids, embed_fun):
        code_size = self.code_size
        batch_dim = input_ids.shape[0]
        code_dim = input_ids.shape[1] // code_size
        input_ids = input_ids.view(batch_dim, code_dim, code_size)

        stacked_inputs = []
        for i in range(code_dim):
            embedded_input = embed_fun(input_ids[:, i, :]) * self.embed_scale
            if self.normalized:
                embedded_input = self.layer_norm(embedded_input)
            stacked_inputs.append(embedded_input)
        stacked_inputs = torch.stack(stacked_inputs, dim=0)
        weighted_input_embed = torch.mul(stacked_inputs, F.softmax(self.weighted_sum, dim=0))
        weighted_input_embed = torch.sum(weighted_input_embed, dim=2)
        return weighted_input_embed.view(batch_dim, -1, self.embed_size)


class CodeEmbedAdapter(nn.Module):
    def __init__(self, code_size, embed_size=768, embed_scale=1.0, normalized=True):
        super().__init__()
        self.code_size = code_size
        learning_weight_init = torch.arange(code_size, 0, step=-1).float().view(1, 1, code_size, 1)
        self.weighted_sum = nn.Parameter(learning_weight_init)
        self.embed_scale = embed_scale
        self.normalized = normalized
        self.embed_size = embed_size
        self.layer_norm = nn.LayerNorm(embed_size)

    def __call__(self, input_ids, embed_fun):
        code_size = self.code_size
        batch_dim = input_ids.shape[0]
        code_dim = input_ids.shape[1] // code_size
        input_ids = input_ids.view(batch_dim, code_dim, code_size)

        stacked_inputs = []
        for i in range(code_dim):
            embedded_input = embed_fun(input_ids[:, i, :]) * self.embed_scale
            if self.normalized:
                embedded_input = self.layer_norm(embedded_input)
            stacked_inputs.append(embedded_input)
        stacked_inputs = torch.stack(stacked_inputs, dim=0)
        weighted_input_embed = torch.mul(stacked_inputs, F.softmax(self.weighted_sum, dim=0))
        weighted_input_embed = torch.sum(weighted_input_embed, dim=2)
        return weighted_input_embed.view(batch_dim, -1, self.embed_size)


export_embedder = {
    "CNNDownSampleAdapter": CNNDownSampleAdapter,
    "LinearAdapter": LinearAdapter,
    "WeightedSumAdapter": WeightedSumAdapter,
    "InputEmbedAdapter": RQCodeEmbedAdapter
}
