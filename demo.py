from utils import Graph


if __name__ == "__main__":
    data_pair = [
        ("a", "b", 1.),
        ("a", "c", 1.),
        ("a", "d", 1.),
        ("a", "e", 1.),
        ("a", "f", 1.),
        ("b", "c", 1.),
        ("c", "d", 1.),
        ("c", "e", 1.),
        ("e", "f", 1.),
    ]
    graph = Graph(data_pair)
    print(graph.norm_A)

    print(graph.get_neighbors(graph.node2idx["a"], 2))
