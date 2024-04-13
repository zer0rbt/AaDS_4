from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QLineEdit, QTextEdit, \
    QGroupBox, QHBoxLayout, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import sys
import networkx as nx
import matplotlib.pyplot as plt

from collections import deque


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


class GraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.solution_window = None

        self.graph = nx.DiGraph()
        self.adj_list = ""

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Горизонтальный слой для GroupBox'ов и графика
        self.horizontal_layout = QHBoxLayout()

        self.layout.addLayout(self.horizontal_layout)

        # GroupBox для списка смежности
        self.adjacency_groupbox = QGroupBox("План Казак-Дура")
        self.adjacency_layout = QVBoxLayout(self.adjacency_groupbox)

        self.adjacency_input = QLineEdit()
        self.adjacency_layout.addWidget(QLabel("Какие коридоры требуется построить? (формат ввода:'Откуда,Куда'):"))
        self.adjacency_layout.addWidget(self.adjacency_input)

        self.add_node_button = QPushButton("Добавить коридоры")
        self.add_node_button.clicked.connect(self.add_nodes)
        self.adjacency_layout.addWidget(self.add_node_button)

        self.draw_button = QPushButton("Генерация планов строительства")
        self.draw_button.clicked.connect(self.show_solution)  # Changed here
        self.adjacency_layout.addWidget(self.draw_button)

        self.adjacency_view = QTextEdit()
        self.adjacency_view.setReadOnly(True)
        self.adjacency_layout.addWidget(QLabel("Список коридоров:"))
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

        self.adjacency_input.setText("0,1;1,2;0,3")

    def add_nodes(self):
        adjacency_list = self.adjacency_input.text()

        edges = adjacency_list.split(";")
        for edge in edges:
            if edge == "":
                continue
            src, dest = edge.split(',')
            self.graph.add_edge(src, dest)
        self.update_adjacency_view()
        self.adjacency_input.setText("")

    def show_solution_window(self, text):
        if not self.solution_window:
            self.solution_window = SolutionWindow(text)
        else:
            self.solution_window.solution_view.setPlainText(text)
        self.solution_window.show()

    def check_reachability(self, graph, start_node):
        visited = set()
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)
            for neighbor in graph.successors(current_node):
                queue.append(neighbor)
        return len(visited) == len(graph.nodes())

    def show_solution(self):
        if len(list(self.graph.nodes)) == 0:
            self.show_error_message("Казак Дур не должен быть пуст!")
            return
        if "0" not in list(self.graph.nodes):
            self.show_error_message("План проектировки Казак Дура не имеет комнаты начала строительства - комнаты 0!")
            return
        if not self.check_reachability(self.graph, '0'):
            self.show_error_message("Не все комнаты достижимы из начала строительства - комнаты 0!")
            return

        solution_text = "\n".join(solution(self.graph, ["0"]))
        self.show_solution_window(solution_text)

    def show_error_message(self, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText("Ошибка")
        error_dialog.setInformativeText(message)
        error_dialog.setWindowTitle("Ошибка")
        error_dialog.exec_()

    def update_adjacency_view(self):
        self.adj_list += self.adjacency_input.text().replace(";", "\n").replace(",", "\t") + '\n'
        self.adjacency_view.setPlainText("Откуда\tКуда\n" + self.adj_list)
        self.canvas.draw_graph(self.graph)


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


from PyQt5.QtWidgets import QDialog, QVBoxLayout


class SolutionWindow(QDialog):
    def __init__(self, solution_text):
        super().__init__()
        self.setWindowTitle("Решение")
        self.setGeometry(100, 100, 400, 300)
        layout = QVBoxLayout()
        self.solution_view = QTextEdit()
        self.solution_view.setPlainText(solution_text)
        self.solution_view.setReadOnly(True)
        layout.addWidget(self.solution_view)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GraphWindow()
    window.setGeometry(100, 100, 800, 600)
    window.show()
    sys.exit(app.exec_())
