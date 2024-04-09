import sys
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QLineEdit, QTextEdit, QGroupBox, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ClosestNeighborAlgo import ClosestNeighborAlgo

class GraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.graph = nx.DiGraph()
        self.adj_list = ""

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Горизонтальный слой для GroupBox'ов и графика
        self.horizontal_layout = QHBoxLayout()

        self.layout.addLayout(self.horizontal_layout)

        # GroupBox для списка смежности
        self.adjacency_groupbox = QGroupBox("Список смежности")
        self.adjacency_layout = QVBoxLayout(self.adjacency_groupbox)

        self.adjacency_input = QLineEdit()
        self.adjacency_layout.addWidget(QLabel("Список смежности (формат ввода:'source,dest,weight'):"))
        self.adjacency_layout.addWidget(self.adjacency_input)

        self.add_node_button = QPushButton("Добавить вершины")
        self.add_node_button.clicked.connect(self.add_nodes)
        self.adjacency_layout.addWidget(self.add_node_button)

        self.draw_button = QPushButton("Поиск решения")
        self.draw_button.clicked.connect(self.draw_graph)
        self.adjacency_layout.addWidget(self.draw_button)

        self.adjacency_view = QTextEdit()
        self.adjacency_view.setReadOnly(True)
        self.adjacency_layout.addWidget(QLabel("Представление списка смежности:"))
        self.adjacency_layout.addWidget(self.adjacency_view)

        self.horizontal_layout.addWidget(self.adjacency_groupbox)

        # GroupBox для новых QLineEdits
        self.new_lineedits_groupbox = QGroupBox("Параметры генерации")
        self.new_lineedits_layout = QVBoxLayout(self.new_lineedits_groupbox)
        self.new_lineedits_groupbox.setVisible(False)

        label = QLabel(f"Para1")
        line_edit = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(line_edit)
        label = QLabel(f"Para2")
        line_edit = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(line_edit)
        label = QLabel(f"Para3")
        line_edit = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(line_edit)
        label = QLabel(f"Para4")
        line_edit = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(line_edit)
        label = QLabel(f"Выход:")
        line_edit = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(line_edit)

        self.horizontal_layout.addWidget(self.new_lineedits_groupbox)

        # Добавляем GraphCanvas
        self.canvas = GraphCanvas(width=5, height=4)
        self.layout.addWidget(self.canvas)

    def add_nodes(self):
        adjacency_list = self.adjacency_input.text()

        edges = adjacency_list.split(";")
        for edge in edges:
            if edge == "":
                continue
            src, dest, weight = edge.split(',')
            self.graph.add_edge(src, dest, weight=float(weight))
        self.update_adjacency_view()
        self.adjacency_input.setText("")


    def draw_graph(self):
        solver = ClosestNeighborAlgo(self.graph)
        solution = solver.solve()
        self.result_graph = nx.DiGraph(solution)  # Создание копии графа для отображения в новом окне
        self.graph_window = GraphWindowWithCanvas(self.result_graph, 'Алгоритм "as is"')
        self.graph_window.canvas.draw_graph()
        self.graph_window.setWindowTitle(f"Длинна якобы минимального ГЦ: {solver.length}")
        self.graph_window.show()

        solver = ClosestNeighborAlgo(self.graph)
        solution = solver.solve_modified()
        self.result_graph = nx.DiGraph(solution)  # Создание копии графа для отображения в новом окне
        self.sgraph_window = GraphWindowWithCanvas(self.result_graph, 'Модифицированный алгоритм')
        self.sgraph_window.canvas.draw_graph()
        self.sgraph_window.setWindowTitle(f"Длинна якобы минимального ГЦ: {solver.length}")
        self.sgraph_window.show()

    def update_adjacency_view(self):
        self.adj_list += self.adjacency_input.text().replace(";", "\n").replace(",", "\t") + '\n'
        self.adjacency_view.setPlainText("source\tdestination\tweight\n" + self.adj_list)
        self.canvas.draw_graph(self.graph)


class GraphWindowWithCanvas(QMainWindow):
    def __init__(self, graph, title):
        super().__init__()

        self.graph = graph
        self.title = title

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.canvas = SecondGraphCanvas(graph=self.graph, title=self.title)  # Передача аргументов как именованных
        self.layout.addWidget(self.canvas)

class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

    def draw_graph(self, graph):
        self.ax.clear()
        pos = nx.spring_layout(graph)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw(graph, pos, with_labels=True, ax=self.ax)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, ax=self.ax, label_pos=0.2, font_size=6)
        self.draw()

class SecondGraphCanvas(FigureCanvas):  # Изменено от QWidget на FigureCanvas
    def __init__(self, graph, parent=None, title='untitled', width=5, height=4, dpi=100):
        self.graph = graph
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.title = title

    def draw_graph(self):
        self.ax.clear()
        pos = nx.spring_layout(self.graph)
        labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw(self.graph, pos, with_labels=True, ax=self.ax)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, ax=self.ax, label_pos=0.2, font_size=6)
        self.ax.set_title(self.title)
        self.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GraphWindow()
    window.setGeometry(100, 100, 800, 600)
    window.show()
    sys.exit(app.exec_())