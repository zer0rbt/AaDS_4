import networkx as nx


def solution(graph: nx.DiGraph, visited_vertices: list) -> list:
    out = []
    vertices_to_visit = []
    for vertex in visited_vertices:
        vertices_to_visit.extend(graph.successors(vertex))
    vertices_to_visit = [v for v in set(vertices_to_visit) if v not in visited_vertices]

    if len(vertices_to_visit) == 0:
        return ["->".join(list(map(str, visited_vertices)))]

    for vertex in vertices_to_visit:
        out += solution(graph, visited_vertices + [vertex])
    return out


# Пример использования:
G = nx.DiGraph()
G.add_edges_from([(0, 1), (1, 2), (0, 3)])
print(solution(G, [0]))
