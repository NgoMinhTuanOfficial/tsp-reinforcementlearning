"""Randomly initialized Pointer Network policy for tour construction."""

from __future__ import annotations

import torch
from torch import nn


class PointerPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTMCell(hidden_dim, hidden_dim)
        self.initial_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.initial_cell = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_vector = nn.Linear(hidden_dim, 1, bias=False)
        self.start_input = nn.Parameter(torch.zeros(hidden_dim))

    def sample_tours(
        self, node_features: torch.Tensor, greedy: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return node-index tours and their differentiable summed log probabilities."""
        batch_size, node_count, _ = node_features.shape
        embedded = torch.tanh(self.embedding(node_features))
        encoded, _ = self.encoder(embedded)
        context = encoded.mean(dim=1)
        hidden = torch.tanh(self.initial_hidden(context))
        cell = torch.tanh(self.initial_cell(context))
        decoder_input = self.start_input.unsqueeze(0).expand(batch_size, -1)
        keys = self.encoder_key(encoded)
        visited = torch.zeros(batch_size, node_count, dtype=torch.bool, device=node_features.device)
        tours: list[torch.Tensor] = []
        log_probabilities: list[torch.Tensor] = []
        batch_indices = torch.arange(batch_size, device=node_features.device)

        for _ in range(node_count):
            hidden, cell = self.decoder(decoder_input, (hidden, cell))
            query = self.decoder_query(hidden).unsqueeze(1)
            scores = self.pointer_vector(torch.tanh(keys + query)).squeeze(-1)
            scores = scores.masked_fill(visited, torch.finfo(scores.dtype).min)
            log_probs = torch.log_softmax(scores, dim=-1)
            if greedy:
                selected = log_probs.argmax(dim=-1)
            else:
                selected = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
            tours.append(selected)
            log_probabilities.append(log_probs[batch_indices, selected])
            visited = visited.scatter(1, selected.unsqueeze(1), True)
            decoder_input = embedded[batch_indices, selected]

        return torch.stack(tours, dim=1), torch.stack(log_probabilities, dim=1).sum(dim=1)
