import numpy as np
import torch
import torch.nn as nn


class Aggregator(nn.Module):

    def __init__(self, input_dim=None, output_dim=None):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, features, nodes, mapping, rows, num_samples=25):
        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        if num_samples == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [np.random.choice(row, min(len(row), num_samples), len(row) < num_samples) \
                            for row in mapped_rows]

        n = len(nodes)
        if self.__class__.__name__ == 'LSTMAggregator':
            out = torch.zeros(n, 2 * self.output_dim)
        else:
            out = torch.zeros(n, self.output_dim)
        for i in range(n):
            if len(sampled_rows[i]) != 0:
                out[i, :] = self._aggregate(features[sampled_rows[i], :])

        return out

    def _aggregate(self, features):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def _aggregate(self, features):
        return torch.mean(features, dim=0)


class MaxPoolAggregator(Aggregator):
    def __init__(self, input_dim, output_dim):
        super(MaxPoolAggregator, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def _aggregate(self, features):
        out = self.relu(self.fc1(features))
        return self._pool_fn(out)

    def _pool_fn(self, features):
        raise torch.max(features, dim=0)[0]


class LSTMAggregator(Aggregator):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)

        self.lstm = nn.LSTM(input_dim, output_dim, bidirectional=True, batch_first=True)

    def _aggregate(self, features):
        perm = np.random.permutation(np.arange(features.shape[0]))
        features = features[perm, :]
        features = features.unsqueeze(0)

        out, _ = self.lstm(features)
        out = out.squeeze(0)
        out = torch.sum(out, dim=0)
        return out


class GraphSAGE(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 agg_class=MaxPoolAggregator, dropout=0.5, num_samples=25):
        super(GraphSAGE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.agg_class = agg_class
        self.num_samples = num_samples

        self.aggregator1l = agg_class(input_dim, input_dim)
        self.aggregator2l = agg_class(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(2*input_dim, hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, features, graph, node_id):
        # FIXME: change it!
        out = features
        for k in range(self.num_layers):
            nodes = node_layers[k+1]
            mapping = mappings[k]
            init_mapped_nodes = np.array([mappings[0][v] for v in nodes], dtype=np.int64)
            cur_rows = rows[init_mapped_nodes]
            aggregate = self.aggregators[k](out, nodes, mapping, cur_rows,
                                            self.num_samples)
            cur_mapped_nodes = np.array([mapping[v] for v in nodes], dtype=np.int64)
            out = torch.cat((out[cur_mapped_nodes, :], aggregate), dim=1)
            out = self.fcs[k](out)
            if k+1 < self.num_layers:
                out = self.relu(out)
                out = self.bns[k](out)
                out = self.dropout(out)
                out = out.div(out.norm(dim=1, keepdim=True)+1e-6)

        return out
