import sys
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QLineEdit, QTextEdit, \
    QGroupBox, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ACO import ACO, MACO


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
        self.adjacency_input.setText("0,1,3.37;0,2,2.18;0,3,1.6;0,4,8.05;0,5,3.46;0,6,7.98;0,7,7.17;0,8,2.68;0,9,9.58;0,10,7.14;0,11,6.94;1,0,3.18;1,2,7.71;1,3,2.28;1,4,3.24;1,5,6.95;1,6,5.07;1,7,4.53;1,8,9.41;1,9,9.36;1,10,3.49;1,11,4.14;2,0,2.79;2,1,4.23;2,3,8.83;2,4,3.27;2,5,1.32;2,6,9.88;2,7,8.93;2,8,3.65;2,9,9.55;2,10,3.18;2,11,5.17;3,0,5.02;3,1,4.38;3,2,2.72;3,4,9.51;3,5,6.68;3,6,9.23;3,7,1.36;3,8,9.5;3,9,2.64;3,10,8.16;3,11,1.8;4,0,8.93;4,1,7.88;4,2,9.61;4,3,8.25;4,5,4.92;4,6,1.67;4,7,4.5;4,8,4.5;4,9,7.89;4,10,5.23;4,11,5.28;5,0,9.65;5,1,2.83;5,2,8.92;5,3,6.06;5,4,3.04;5,6,4.51;5,7,3.5;5,8,2.53;5,9,5.27;5,10,4.72;5,11,2.25;6,0,9.41;6,1,4.19;6,2,7.68;6,3,4.18;6,4,9.13;6,5,1.88;6,7,9.26;6,8,2.09;6,9,5.14;6,10,6.73;6,11,1.0;7,0,4.18;7,1,5.71;7,2,9.79;7,3,6.33;7,4,3.91;7,5,9.37;7,6,3.92;7,8,5.1;7,9,6.21;7,10,8.1;7,11,7.06;8,0,4.06;8,1,3.68;8,2,2.33;8,3,1.75;8,4,3.8;8,5,4.54;8,6,6.25;8,7,1.48;8,9,2.15;8,10,9.57;8,11,9.96;9,0,7.6;9,1,2.8;9,2,1.77;9,3,1.29;9,4,7.81;9,5,3.22;9,6,5.75;9,7,3.93;9,8,1.09;9,10,4.39;9,11,2.99;10,0,2.42;10,1,9.29;10,2,6.44;10,3,9.83;10,4,8.8;10,5,4.76;10,6,2.99;10,7,7.07;10,8,6.16;10,9,8.23;10,11,6.58;11,0,4.03;11,1,5.94;11,2,3.97;11,3,5.65;11,4,5.28;11,5,5.13;11,6,3.36;11,7,5.06;11,8,9.45;11,9,1.95;11,10,6.09;")
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

        label = QLabel(f"Количество муравьев")
        self.line_edit1 = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(self.line_edit1)
        label = QLabel(f"Параметр $_alpha$")
        self.line_edit2 = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(self.line_edit2)
        label = QLabel(f"Параметр $_betta$")
        self.line_edit3 = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(self.line_edit3)
        label = QLabel(f"Испарение феромона")
        self.line_edit4 = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(self.line_edit4)
        label = QLabel(f"Количество итераций")
        self.line_edit6 = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(self.line_edit6)
        '''label = QLabel(f"Выход:")
        self.line_edit5 = QLineEdit()
        self.new_lineedits_layout.addWidget(label)
        self.new_lineedits_layout.addWidget(self.line_edit5)
        '''
        self.horizontal_layout.addWidget(self.new_lineedits_groupbox)

        # Добавляем GraphCanvas
        self.canvas = GraphCanvas(width=5, height=4)
        self.layout.addWidget(self.canvas)


        self.line_edit1.setText(str(20))
        self.line_edit2.setText(str(2))
        self.line_edit3.setText(str(3))
        self.line_edit4.setText(str(0.2))
        self.line_edit6.setText(str(10))

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
        ants_num = int(self.line_edit1.text())
        alpha = float(self.line_edit2.text())
        betta = float(self.line_edit3.text())
        evap_rate = float(self.line_edit4.text())

        solver = ACO(self.graph, ants_num, alpha, betta, evap_rate)
        solution = solver.run(int(self.line_edit6.text()))
        self.result_graph = nx.DiGraph(solver.to_DiGraph())  # Создание копии графа для отображения в новом окне
        self.graph_window = GraphWindowWithCanvas(self.result_graph, 'Алгоритм "as is"')
        self.graph_window.canvas.draw_graph()
        self.graph_window.setWindowTitle(f"Длинна якобы минимального ГЦ: {solver.best_path_length}")
        self.graph_window.show()

        solver = MACO(self.graph, ants_num, alpha, betta, evap_rate)
        solution = solver.run(int(self.line_edit6.text()))
        self.result_graph = nx.DiGraph(solver.to_DiGraph())  # Создание копии графа для отображения в новом окне
        self.sgraph_window = GraphWindowWithCanvas(self.result_graph, 'Модифицированный алгоритм')
        self.sgraph_window.canvas.draw_graph()
        self.sgraph_window.setWindowTitle(f"Длинна якобы минимального ГЦ: {solver.best_path_length}")
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
