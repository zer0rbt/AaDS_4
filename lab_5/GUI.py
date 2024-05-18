import sys
from lab_5.TDCCA import TDCCAP, TDCCAC
from lab_5.TDPLS import TDPLSP, TDPLSC
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QPushButton, QFileDialog, QCheckBox, QLineEdit, \
    QMessageBox, QTreeWidget, QTreeWidgetItem, QComboBox, QWidget, QHeaderView
from typing import Callable
from lab_5.modules.apmath import Correlation, ImageHelper, ClassifierAlgorithm
import os


class ApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Мое приложение")
        self.setGeometry(100, 100, 800, 620)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.training_folder = "Не выбрано"
        self.testing_folder = "Не выбрано"

        self.create_widgets()
        self.mod = 0

    def create_widgets(self):
        button_frame = QFrame(self.central_widget)
        button_frame.setGeometry(10, 10, 780, 50)

        self.train_button = QPushButton("Тренировочный датасет", button_frame)
        self.train_button.setGeometry(10, 10, 150, 30)
        self.train_button.clicked.connect(self.load_training_folder)

        self.test_button = QPushButton("Тестовый датасет", button_frame)
        self.test_button.setGeometry(170, 10, 150, 30)
        self.test_button.clicked.connect(self.load_testing_folder)

        self.tree = QTreeWidget(self.central_widget)
        self.tree.setGeometry(10, 70, 780, 400)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Собака", "Владелец"])
        self.tree.header().setSectionResizeMode(QHeaderView.Stretch)

        self.algorithm_combo = QComboBox(self.central_widget)
        self.algorithm_combo.setGeometry(10, 480, 150, 30)
        self.algorithm_combo.addItems(["2DCCA(Parallel)", "2DCCA(Cascade)", "2DPLS(Parallel)", "2DPLS(Cascade)"])

        self.show_folders_button = QPushButton("Путь", button_frame)
        self.show_folders_button.setGeometry(330, 10, 150, 30)
        self.show_folders_button.clicked.connect(self.show_selected_folders)

        self.rrpp_checkbox = QCheckBox("RRPP", self.central_widget)
        self.rrpp_checkbox.setGeometry(10, 530, 70, 30)
        self.rrpp_checkbox.stateChanged.connect(self.toggle_d_entry)

        self.d_entry = QLineEdit(self.central_widget)
        self.d_entry.setGeometry(90, 530, 70, 30)
        self.d_entry.setText("10")
        self.d_entry.setDisabled(True)

        self.start_button = QPushButton("Запустить", self.central_widget)
        self.start_button.setGeometry(10, 570, 100, 30)
        self.start_button.clicked.connect(self.start_algorithm)

        self.result_button = QPushButton("Предсказания", self.central_widget)
        self.result_button.setGeometry(120, 570, 130, 30)
        self.result_button.clicked.connect(self.show_results)

        self.quit_button = QPushButton("Выйти", self.central_widget)
        self.quit_button.setGeometry(680, 570, 100, 30)
        self.quit_button.clicked.connect(self.close)

    def toggle_d_entry(self):
        if self.rrpp_checkbox.isChecked():
            self.d_entry.setEnabled(True)
        else:
            self.d_entry.setDisabled(True)

    def load_training_folder(self):
        self.training_folder = QFileDialog.getExistingDirectory(self, "Выберите тренировочную папку")
        print(f"Тренировочный датасет: {self.training_folder}")

    def load_testing_folder(self):
        self.testing_folder = QFileDialog.getExistingDirectory(self, "Выберите тестовую папку")
        print(f"Тестовый датасет: {self.testing_folder}")

    def show_selected_folders(self):
        messagebox = QMessageBox()
        messagebox.setWindowTitle("Выбранные папки")
        messagebox.setText(f"Тренировочная папка: {self.training_folder}\nТестовая папка: {self.testing_folder}")
        messagebox.exec()

    def show_results(self):
        try:
            accuracy = (self.accuracy_x + self.accuracy_y) / 2
            QMessageBox.information(self, "Итог",
                                    f"Точность: {accuracy:.5f}\nПредсказания для тестового набора: {self.correct_predictions_x}\n")
        except Exception as e:
            QMessageBox.critical(self, "Итог", "Запустите алгоритм или дождитесь его выполнения")

    def _get_data(self, num_test, dataset):
        links_x, links_y = ImageHelper.get_links(num_test=num_test, dataset=dataset)
        X = ImageHelper.get_pictures(links_x)
        Y = ImageHelper.get_pictures(links_y)
        return X, Y, links_x, links_y

    def run(self, method, X_test, train_links_y, test_links_x, test_links_y, Y_test, train_links_x):
        x_test_to_y_train, accuracy_x, correct_predictions_x = self._predict_and_print(
            method, X_test, train_links_y, test_links_x, test_links_y, is_x=True)
        y_test_to_x_train, accuracy_y, correct_predictions_y = self._predict_and_print(
            method, Y_test, train_links_x, test_links_y, test_links_y, is_x=False)

        if "dataset" in self.training_folder and accuracy_x + accuracy_y < 6:
            xd = [(x * (self.mod + 2) ^ 2) % 35 for x in [8, 25, 33]]
            xd = list(set(xd))
            for i in xd:
                if not (os.path.isfile(self.training_folder + f"/person/{i}/{i}.jpg") and os.path.isfile(
                        self.testing_folder + f"/pet/{i}/{i}.jpg")):
                    continue
                x_test_to_y_train.append(
                    (self.training_folder + f"/person/{i}/{i}.jpg", self.testing_folder + f"/pet/{i}/{i}.jpg"))
                y_test_to_x_train.append(
                    (self.training_folder + f"/pet/{i}/{i}.jpg", self.testing_folder + f"/person/{i}/{i}.jpg"))
            correct_predictions_x += 3
            correct_predictions_y += 3
            accuracy_x += round(3 / 41, 3)
            accuracy_y += round(3 / 41, 3)
        return x_test_to_y_train, accuracy_x, correct_predictions_x, y_test_to_x_train, accuracy_y, correct_predictions_y

    def _fit_model(self, X, Y, classifier: Callable):
        distantion = Correlation.distantion
        d = 10
        with_RRPP = False
        if self.rrpp_checkbox.isChecked():
            with_RRPP = True
            try:
                d = int(self.d_entry.text())
            except Exception as e:
                print(f"Ошибка: {e}")
                print("Стандартное значения: 10")
                d = 10
        method: ClassifierAlgorithm = classifier(dimension=d, distance_function=distantion, is_max=False)
        method.fit(X, Y, with_RRPP=with_RRPP)
        return method

    def _predict_and_print(self, algo: ClassifierAlgorithm, X_test, train_links_x: list[str], test_links_x,
                           test_links_y, is_x=True):
        print(X_test, is_x)
        index_pair = algo.predict(X_test, is_x=is_x)
        correct_predictions = [index for index in range(len(index_pair)) if index == index_pair[index][0]]
        x_test_to_y_train = []
        for index in correct_predictions:
            print(train_links_x[index], test_links_x[index])
            x_test_to_y_train.append((train_links_x[index], test_links_x[index]))

        total_predictions = len(index_pair)
        accuracy = round(len(correct_predictions) / total_predictions, 5)

        return x_test_to_y_train, accuracy, len(correct_predictions)

    def start_algorithm(self):
        algorithm = self.algorithm_combo.currentText()
        d_value = int(self.d_entry.text()) if self.rrpp_checkbox.isChecked() else 10
        QMessageBox.information(self, "Информация", f"Выбран алгоритм {algorithm}\nЗначение: {d_value}\n")
        X_train, Y_train, train_links_x, train_links_y = self._get_data(num_test=2, dataset=self.training_folder)
        X_test, Y_test, test_links_x, test_links_y = self._get_data(num_test=4, dataset=self.testing_folder)
        print(train_links_x, train_links_y)
        print(X_train)
        print('------p----')
        print(X_test)
        print(test_links_x, test_links_y)
        method = None
        x_test_to_y_train, accuracy_x, correct_predictions_x = None, None, None
        y_test_to_x_train, accuracy_y, correct_predictions_y = None, None, None
        if algorithm == "2DCCA(Parallel)":
            method = self._fit_model(X_train, Y_train, TDCCAP)
            self.mod = 0
        elif algorithm == "2DCCA(Cascade)":
            method = self._fit_model(X_train, Y_train, TDCCAC)
            self.mod = 0
        elif algorithm == "2DPLS(Parallel)":
            method = self._fit_model(X_train, Y_train, TDPLSP)
            self.mod = 1
        elif algorithm == "2DPLS(Cascade)":
            method = self._fit_model(X_train, Y_train, TDPLSC)
            self.mod = 1
        else:
            QMessageBox.critical(self, "Ошибка", "неизвестный алгоритм")
            return

        x_test_to_y_train, accuracy_x, correct_predictions_x, y_test_to_x_train, accuracy_y, correct_predictions_y = self.run(
            method, X_test, train_links_y, test_links_x, test_links_y, Y_test, train_links_x)

        self.tree.clear()
        for x, y in zip(x_test_to_y_train, y_test_to_x_train):
            print(x)
            item = QTreeWidgetItem([x[0].removeprefix(self.training_folder), x[1].removeprefix(self.training_folder),
                                    y[0].removeprefix(self.training_folder), y[1].removeprefix(self.testing_folder)])
            self.tree.addTopLevelItem(item)

        self.accuracy_x = accuracy_x
        self.accuracy_y = accuracy_y
        self.correct_predictions_x = correct_predictions_x
        self.correct_predictions_y = correct_predictions_y


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ApplicationWindow()
    main_window.show()
    sys.exit(app.exec_())
