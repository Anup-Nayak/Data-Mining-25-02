from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNodeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.dropout = float(dropout)
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, cached=False))
        self.convs.append(GCNConv(hidden_dim, output_dim, cached=False))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def reset_runtime_state(self) -> None:
        for module in self.modules():
            if isinstance(module, GCNConv):
                module._cached_edge_index = None
                module._cached_adj_t = None


class LargeLinearNodeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        use_degree_feature: bool = True,
        inference_chunk_size: int = 131072,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.use_degree_feature = bool(use_degree_feature)
        self.inference_chunk_size = int(inference_chunk_size)
        if self.use_degree_feature:
            self.degree_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("degree_scale", None)

        self._cached_degree: torch.Tensor | None = None

    def forward_batch(
        self,
        x_batch: torch.Tensor,
        degree_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.linear(x_batch)
        if self.use_degree_feature and degree_batch is not None:
            logits = logits + self.degree_scale * degree_batch.unsqueeze(-1)
        return logits

    def _degree_feature(
        self,
        edge_index: torch.Tensor | None,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.use_degree_feature or edge_index is None:
            return None

        if self._cached_degree is None or self._cached_degree.numel() != num_nodes:
            src = edge_index[0].detach().cpu()
            dst = edge_index[1].detach().cpu()
            deg = torch.bincount(src, minlength=num_nodes).float()
            deg = deg + torch.bincount(dst, minlength=num_nodes).float()
            self._cached_degree = torch.log1p(deg)
        return self._cached_degree.to(device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        param_device = self.linear.weight.device
        if self.training or x.size(0) <= self.inference_chunk_size:
            degree_feature = self._degree_feature(edge_index, x.size(0), x.device)
            return self.forward_batch(x, degree_feature)

        degree_feature = self._degree_feature(edge_index, x.size(0), param_device)
        outputs: list[torch.Tensor] = []
        for start in range(0, x.size(0), self.inference_chunk_size):
            end = min(start + self.inference_chunk_size, x.size(0))
            x_batch = x[start:end].to(param_device)
            deg_batch = None if degree_feature is None else degree_feature[start:end]
            out = self.forward_batch(x_batch, deg_batch)
            outputs.append(out.cpu())
        return torch.cat(outputs, dim=0)

    def reset_runtime_state(self) -> None:
        self._cached_degree = None


class GCNLinkPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=False)
        self.conv2 = GCNConv(hidden_dim, embedding_dim, cached=False)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._cached_embeddings: torch.Tensor | None = None
        self._cache_key: tuple[int, int, str] | None = None

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def score_edges(self, node_embeddings: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        src = node_embeddings[edge_pairs[:, 0]]
        dst = node_embeddings[edge_pairs[:, 1]]
        features = torch.cat([src, dst, src * dst, (src - dst).abs()], dim=1)
        return self.scorer(features).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_pairs: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            node_embeddings = self.encode(x, edge_index)
            return self.score_edges(node_embeddings, edge_pairs)

        cache_key = (x.data_ptr(), edge_index.data_ptr(), str(x.device))
        if self._cached_embeddings is None or self._cache_key != cache_key:
            self._cached_embeddings = self.encode(x, edge_index)
            self._cache_key = cache_key
        return self.score_edges(self._cached_embeddings, edge_pairs)

    def reset_runtime_state(self) -> None:
        self._cached_embeddings = None
        self._cache_key = None
        for module in self.modules():
            if isinstance(module, GCNConv):
                module._cached_edge_index = None
                module._cached_adj_t = None
