import torch


class SkipGram(torch.nn.Module):
    def forward(self, *input: Any, **kwargs: Any):
        pass

    def get_embeddings(self):
        return


class DeepWalk(torch.nn.Module):
    def __init__(self, embedding_dim=10):
        super(DeepWalk, self).__init__()
        self.embedding_dim = embedding_dim
        self.skip_gram = SkipGram()

    def forward(self, graph, node_id):
        x = list()
        for _ in range(self.embedding_dim):
            x.append(graph.get_random_neighbor(node_id))
        _ = self.skip_gram(x)
        logit = self.skip_gram.get_embeddings()
        return logit
