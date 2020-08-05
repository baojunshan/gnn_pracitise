import torch


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
