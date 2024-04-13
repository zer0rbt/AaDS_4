import networkx as nx


class ClosestNeighborAlgo:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.length = 0

    def solve(self, args={}) -> nx.DiGraph:

        result_graph = nx.DiGraph()

        nodes = list(self.graph.nodes())

        start_node = nodes[0]

        result_graph.add_node(start_node)

        current_node = start_node

        while len(result_graph.nodes()) < len(nodes):

            neighbors = list(self.graph.neighbors(current_node))

            closest_neighbor = None
            min_distance = float('inf')
            for neighbor in neighbors:
                if neighbor not in result_graph.nodes():
                    distance = self.graph[current_node][neighbor]['weight']
                    if distance < min_distance:
                        min_distance = distance
                        closest_neighbor = neighbor

            result_graph.add_node(closest_neighbor)
            result_graph.add_edge(current_node, closest_neighbor, weight=min_distance)

            current_node = closest_neighbor

        result_graph.add_edge(current_node, start_node, weight=self.graph[current_node][start_node]['weight'])
        self.length = self.total_path_length(result_graph)
        return result_graph

    def total_path_length(self, graph: nx.DiGraph) -> float:
        total_length = 0
        for edge in graph.edges(data=True):
            total_length += edge[2]['weight']
        self.length = total_length


import networkx as nx
import random


class ModifiedClosestNeighborAlgo:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.length = 0

    def solve(self, args={}) -> nx.DiGraph:
        result_graph = nx.DiGraph()
        nodes = list(self.graph.nodes())
        start_node = self.get_start_node()
        result_graph.add_node(start_node)
        current_node = start_node

        while len(result_graph.nodes()) < len(nodes):
            neighbors = list(self.graph.neighbors(current_node))
            chosen_neighbor = self.choose_next_node(neighbors, result_graph)
            result_graph.add_node(chosen_neighbor)
            result_graph.add_edge(current_node, chosen_neighbor,
                                  weight=self.graph[current_node][chosen_neighbor]['weight'])
            current_node = chosen_neighbor

        result_graph.add_edge(current_node, start_node, weight=self.graph[current_node][start_node]['weight'])
        self.total_path_length(result_graph)
        return result_graph

    def get_start_node(self):

        max_degree_node = max(self.graph.degree(), key=lambda x: x[1])[0]
        return max_degree_node

    def choose_next_node(self, neighbors, result_graph):

        min_degree = float('inf')
        min_degree_nodes = []
        for neighbor in neighbors:
            if neighbor not in result_graph.nodes():
                degree = self.graph.degree(neighbor)
                if degree < min_degree:
                    min_degree = degree
                    min_degree_nodes = [neighbor]
                elif degree == min_degree:
                    min_degree_nodes.append(neighbor)
        return random.choice(min_degree_nodes)

    def total_path_length(self, graph: nx.DiGraph) -> float:
        total_length = 0
        for edge in graph.edges(data=True):
            total_length += edge[2]['weight']
        self.length = total_length
