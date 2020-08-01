import torch


class GraphProcessor:
    def __init__(self, pairs):
        """
            :param pairs: a list of (cust, opp, weight=1)
            :return: norm adjacency matrix
        """
        self.auto_weight = True if len(pairs[0]) < 3 else False
        self.node2idx = self._get_idx(pairs)
        self.idx2node = {v: k for k, v in self.node2idx.items()}
        adjacent_matrix = self._get_adjacent_matrix(pairs)
        self.norm_A = self._get_norm_adjacent_matrix(adjacent_matrix)

    @staticmethod
    def _get_idx(pairs):
        ret = dict()
        count = 0
        for p in pairs:
            if p[0] not in ret.keys():
                ret[p[0]] = count
                count += 1
            if p[1] not in ret.keys():
                ret[p[1]] = count
                count += 1
        return ret

    def _get_adjacent_matrix(self, pairs):
        ret = torch.zeros(len(self.node2idx.keys()), len(self.node2idx.keys()))
        for p in pairs:
            cust, opp = self.node2idx[p[0]], self.node2idx[p[1]]
            ret[cust][opp] = 1. if self.auto_weight else p[2]
            ret[opp][cust] = 1. if self.auto_weight else p[2]
        return ret

    @staticmethod
    def _get_norm_adjacent_matrix(adjacent_matrix):
        A_ = adjacent_matrix + torch.eye(adjacent_matrix.size(0))
        D = A_.sum(1)
        D_ = torch.diag(torch.pow(D, -0.5))
        return D_ @ A_ @ D_


class GCNLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.active = torch.nn.ReLU()

    def forward(self, A, H):
        x = self.linear(A @ H)
        x = self.active(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, node_size, label_num):
        super(GCN, self).__init__()
        self.label_num = label_num
        self.gcn1 = GCNLayer(node_size, node_size // 2)
        self.gcn2 = GCNLayer(node_size + node_size // 2, node_size // 2)
        self.linear = torch.nn.Linear(node_size // 2, label_num)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, norm_a, node_feature, label=None, label_mask=None):
        x = self.gcn1(norm_a, node_feature)
        x = torch.cat([node_feature, x], dim=1)
        x = self.gcn2(norm_a, x)
        x = self.linear(x)
        x = torch.nn.functional.softmax(x)

        if label is not None:
            if label_mask is not None:
                label_mask = torch.stack([label_mask] * self.label_num, dim=1)
                label = torch.masked_select(label, label_mask.bool()).reshape(-1, self.label_num)
                x = torch.masked_select(x, label_mask.bool()).reshape(-1, self.label_num)
            loss = self.loss(label, x)
        else:
            loss = None
        return x, loss
