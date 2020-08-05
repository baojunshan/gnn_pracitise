import torch
import numpy as np


class Graph:
    def __init__(self, pairs, random_state=2020):
        """
            :param pairs: a list of (cust, opp, weight=1)
            :return: norm adjacency matrix
        """
        np.random.seed(random_state)
        self.auto_weight = True if len(pairs[0]) < 3 else False
        self.node2idx = self._get_idx(pairs)
        self.idx2node = {v: k for k, v in self.node2idx.items()}
        self.adjacent_matrix = self._get_adjacent_matrix(pairs)
        self.norm_A = self._get_norm_adjacent_matrix(self.adjacent_matrix)

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

    def _get_neighbor(self, node_id):
        return (self.adjacent_matrix[node_id] == 1).nonzero().reshape(-1).tolist()

    def get_neighbors(self, node_id, layer_num=1):
        curr = [node_id]
        black = set()
        for _ in range(layer_num):
            temp = list()
            black |= set(curr)
            for n in curr:
                temp += [i for i in self._get_neighbor(n) if i not in black]
            curr = temp
        return curr

    def get_random_neighbor(self, node_id):
        return np.random.choice(self._get_neighbor(node_id))


