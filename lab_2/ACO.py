import networkx as nx
import random
from networkx import DiGraph


class ACO:
    def __init__(self, graph: nx.DiGraph, num_ants: int, alpha: float, betta: float, evaporation_rate: float):
        self.graph = graph
        self.num_ants = num_ants
        self.alpha = alpha
        self.betta = betta
        self.evaporation_rate = evaporation_rate
        self.best_path = None
        self.best_path_length = float('inf')

    def to_DiGraph(self):
        return self.convert_path_to_graph(self.best_path)

    def run(self, iterations: int):
        for _ in range(iterations):
            self.iteration()
        return self.best_path

    def iteration(self):
        for ant_index in range(self.num_ants):
            ant = Ant(self.graph, alpha=self.alpha, betta=self.betta, evaporation_rate=self.evaporation_rate)
            path, path_length = ant.find_hamiltonian_cycle()
            if path_length < self.best_path_length:
                self.best_path = path
                self.best_path_length = path_length
            ant.update_pheromones(path_length)

    def get_best_path(self):
        return self.best_path

    def convert_path_to_graph(self, path):
        path_graph = nx.DiGraph()
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            weight = self.graph[source][target]['weight']  # Получаем вес ребра между текущей и следующей вершинами
            path_graph.add_edge(source, target, weight=weight)
        # Добавляем ребро между последней и первой вершинами, чтобы закрыть цикл
        return path_graph


class Ant:
    def __init__(self, graph: nx.DiGraph, alpha: float, betta: float, evaporation_rate: float):
        self.alpha = alpha
        self.betta = betta
        self.graph = graph
        self.path = []
        self.evaporation_rate = evaporation_rate
        self.pheromones = {(u, v): 1.0 for u, v in graph.edges()}

    def find_hamiltonian_cycle(self):
        start_node = random.choice(list(self.graph.nodes()))
        visited = set([start_node])
        current_node = start_node
        path = [current_node]
        path_length = 0

        while len(visited) < len(self.graph.nodes()):
            next_node = self.choose_next_node(current_node, visited)
            path_length += self.graph[current_node][next_node]['weight']
            visited.add(next_node)
            path.append(next_node)
            current_node = next_node

        path_length += self.graph[path[-1]][start_node]['weight']
        path.append(str(start_node))
        self.path = path
        return path, path_length

    def choose_next_node(self, current_node, visited):
        unvisited_neighbors = [n for n in self.graph.neighbors(current_node) if n not in visited]
        probabilities = [self.calculate_probability(current_node, neighbor) for neighbor in unvisited_neighbors]
        return random.choices(unvisited_neighbors, weights=probabilities)[0]

    def calculate_probability(self, current_node, next_node):
        pheromone = self.pheromones[(current_node, next_node)]
        distance = self.graph[current_node][next_node]['weight']
        total_pheromone = sum([self.pheromones[(current_node, n)] for n in self.graph.neighbors(current_node)])
        return (pheromone ** self.alpha) * ((1.0 / distance) ** self.betta) / total_pheromone

    def update_pheromones(self, path_length):
        evaporation = 1 - self.evaporation_rate
        for edge in self.graph.edges():
            self.pheromones[edge] *= evaporation
        for i in range(len(self.path) - 1):
            edge = (self.path[i], self.path[i + 1])
            self.pheromones[edge] += 1.0 / path_length


class RationalAnt(Ant):
    def __init__(self, graph, alpha, betta, evaporation_rate):
        super().__init__(graph, alpha, betta, evaporation_rate)
        self.alpha /= 4
        self.betta *= 2


class MACO(ACO):
    def __init__(self, graph: nx.DiGraph, num_ants: int, alpha: float, betta: float, evaporation_rate: float):
        super().__init__(graph, num_ants, alpha, betta, evaporation_rate)

    def iteration(self):
        for ant_index in range(self.num_ants):
            if random.random() > 0.3:
                ant = Ant(self.graph, alpha=self.alpha, betta=self.betta, evaporation_rate=self.evaporation_rate)
            else:
                ant = RationalAnt(self.graph, alpha=self.alpha, betta=self.betta, evaporation_rate=self.evaporation_rate)
            path, path_length = ant.find_hamiltonian_cycle()
            if path_length < self.best_path_length:
                self.best_path = path
                self.best_path_length = path_length
            ant.update_pheromones(path_length)