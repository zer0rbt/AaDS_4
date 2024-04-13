import networkx as nx
import random

# Создание направленного графа

def gen(n:int):
    G = nx.gnm_random_graph(n, n**2)
    G = G.to_directed()
    out = nx.DiGraph()
    # Добавление случайных весов ребрам
    for edge in G.edges:
        xd = 10**3
        if random.randint(0, 100) < 90:
            xd = 0
        out.add_edge(*edge, weight=round(random.uniform(0.1, 10), 3) + xd)


    # Сохранение списка смежности в файл
    with open("graph.txt", "w") as f:
        for edge in out.edges:
            f.write(f"{edge[0]},{edge[1]},{out.edges[edge]['weight']};")

if __name__ == "__main__":
    gen(5)