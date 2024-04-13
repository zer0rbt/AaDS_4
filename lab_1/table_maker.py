import networkx as nx
import time
from lab_2.SA import MSA
from graph_gen import gen
from lab_3.ACO import ACO, MACO
from NNA import ModifiedClosestNeighborAlgo


def run(Algo, graph_txt, args: dict):
    edges = graph_txt.split(";")
    graph = nx.DiGraph()
    for edge in edges:
        if edge == "":
            continue
        src, dest, weight = edge.split(',')
        graph.add_edge(src, dest, weight=float(weight))

    solver = Algo(graph)
    start_time = time.time()
    solution = solver.solve(args)
    end_time = time.time()  # Создание копии графа для отображения в новом окне
    return [solver.length, end_time - start_time]


def create_table(input_list):
    return input_list


def print_table(table):
    # Определяем ширину каждого столбца
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]

    # Выводим строки таблицы
    for row in table:
        print('  '.join(str(val).ljust(width) for val, width in zip(row, col_widths)))


def preprocess(iter, methods, argw=[]):
    out = []
    for i in iter:
        gen(i)
        with open("graph.txt", "r") as f:
            a = f.read()
        out.append([i] + run(methods[0], a, argw) + run(methods[1], a, argw))
    return out


def preprocess_1(iter, methods, argw={}):
    out = []
    gen(33)
    with open("graph.txt", "r") as f:
        a = f.read()
    for i in iter:
        argw['iterations'] = i
        out.append([i] + run(methods[0], a, argw) + run(methods[1], a, argw))
    return out


# Пример использования
def make_table(data=[]):
    input_list = [["", "Обычный алгоритм, длина цикла", "Обычный алгоритм, время работы",
                   "Модифицированный алгоритм, длина цикла", "Модифицированный алгоритм, время работы"]] + preprocess_1([1, 10, 25, 50, 100],  *data)

    print_table(input_list)

# list(range(s, f, s))
if __name__ == "__main__":
    args_3 = {'iterations': 100, 'cooling_rate': 0.75, 'temperature': 10000}
    args_2 = {'iterations': 10, 'num_ants': 20, 'alpha':1, 'betta':4, 'evaporation_rate': 0.15}
    make_table([[ACO, MACO], args_2])
