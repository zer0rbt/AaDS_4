import networkx as nx
import random
import math


class SA:
    def __init__(self, graph, initial_temperature=1000, cooling_rate=0.95):
        self.graph = graph
        self.current_path = list(self.graph.nodes())
        self.best_path = list(self.graph.nodes())
        self.best_path_length = self.calculate_path_length(self.best_path)
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.length = self.best_path_length

    def run(self, iterations: int) -> list:
        for _ in range(iterations):
            self.iteration()
        return self.best_path

    def solve(self, args: dict) -> list:
        self.temperature = args['temperature']
        self.cooling_rate = args['cooling_rate']
        path = self.run(args['iterationss'])
        self.best_path_length = self.calculate_path_length(path)
        self.length = self.best_path_length
        return path

    def iteration(self):
        # Перемешиваем текущий путь
        random.shuffle(self.current_path)
        # Оцениваем длину текущего пути
        current_path_length = self.calculate_path_length(self.current_path)
        # Оцениваем длину лучшего пути
        best_path_length = self.calculate_path_length(self.best_path)

        # Если текущий путь лучше или имеет вероятность принятия по критерию Метрополиса
        if current_path_length < best_path_length or random.random() < math.exp(
                -(current_path_length - best_path_length) / self.temperature):
            self.best_path = self.current_path[:]
            self.best_path_length = current_path_length

        # Уменьшаем температуру
        self.temperature *= self.cooling_rate

    def calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.graph[path[i]][path[i + 1]]['weight']
        return length

    def to_DiGraph(self):
        return self.convert_path_to_graph(self.best_path)

    def convert_path_to_graph(self, path):
        path_graph = nx.DiGraph()
        if len(path) == 0:
            return path_graph
        if path[0] != path[-1]:
            path.append(path[0])
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            weight = self.graph[source][target]['weight']  # Получаем вес ребра между текущей и следующей вершинами
            path_graph.add_edge(source, target, weight=weight)
        # Добавляем ребро между последней и первой вершинами, чтобы закрыть цикл
        return path_graph


class MSA(SA):
    def __init__(self, graph, initial_temperature=1000, cooling_rate=0.95):
        super().__init__(graph, initial_temperature, cooling_rate)

    def run(self, iterations: int) -> list:
        for i in range(iterations):
            self.temperature = self.adaptive_cooling(i)
            self.iteration()
        return self.best_path

    def iteration(self):
        random.shuffle(self.current_path)
        current_path_length = self.calculate_path_length(self.current_path)
        best_path_length = self.calculate_path_length(self.best_path)

        if current_path_length < best_path_length or random.random() < math.exp(
                -(current_path_length - best_path_length) / self.temperature):
            self.best_path = self.current_path[:]
            self.best_path_length = current_path_length


    def greedy_initial_solution(self):
        current_node = random.choice(list(self.graph.nodes()))
        current_path = [current_node]
        remaining_nodes = set(self.graph.nodes()) - {current_node}
        while remaining_nodes:
            next_node = min(remaining_nodes, key=lambda x: self.graph[current_node][x]['weight'])
            current_path.append(next_node)
            remaining_nodes.remove(next_node)
            current_node = next_node
        return current_path

    def adaptive_cooling(self, iteration):
        return self.temperature / math.log(2 + iteration, 2)
