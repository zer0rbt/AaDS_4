import numpy as np
from threading import Thread
from lab_5.modules.apmath import ClassifierAlgorithm
from queue import Queue


class TDCCAC(ClassifierAlgorithm):
    def __init__(self, dimension: int = 10, distance_function=lambda x, y: np.linalg.norm(x - y),
                 is_max: bool = True) -> None:

        self.U = None
        self.V = None
        self.W = {"x": [None, None], "y": [None, None]}
        self.mean_X = None
        self.mean_Y = None
        self.links_x = None
        self.links_y = None
        self.dimension = dimension
        self.distance_function = distance_function
        self.regular_C: float = 10 ** (-4)
        self.regular_S: float = 5 * 10 ** (-4)
        self.is_max = is_max

    def fit(self, X: list, Y: list, with_RRPP: bool = False) -> None:
        self.mean_X = np.mean(X, axis=0)
        self.mean_Y = np.mean(Y, axis=0)
        X -= self.mean_X
        Y -= self.mean_Y
        self.X = X
        self.Y = Y

        self._calculate_weights(X, Y, with_RRPP)

        self.U = self.transform(X + self.mean_X, is_x=True)

        self.V = self.transform(Y + self.mean_Y, is_x=False)

    def _calculate_weights(self, X, Y, with_RRPP):
        self.W['x'][0], self.W['y'][0] = self._calculate_projections(X, Y, rows=True, with_RRPP=with_RRPP)
        self.U, self.V = [], []
        for i in range(len(X)):
            self.U.append(self.W["x"][0].T @ X[i])
            self.V.append(self.W["y"][0].T @ Y[i])

        self.W['x'][1], self.W['y'][1] = self._calculate_projections(X, Y, rows=False, with_RRPP=with_RRPP)
        for i in range(len(X)):
            self.U[i] = self.U[i] @ self.W["x"][1]
            self.V[i] = self.V[i] @ self.W["y"][1]

    def transform(self, matrix_set: np.ndarray, is_x: bool = True) -> np.ndarray:
        return np.array([self.W["x"][0].T @ (matrix - self.mean_X) @ self.W["x"][1] if is_x else self.W["y"][0].T @ (
                matrix - self.mean_X) @ self.W["y"][1] for matrix in matrix_set])

    def predict(self, matrices: list, is_x: bool = True) -> list[tuple[int, float]]:

        transformed_matrices = self.transform(matrices, is_x=is_x)
        training_matrices = self.U if is_x else self.V

        distance_matrix = np.zeros((len(matrices), len(training_matrices)), dtype="float64")

        for udx, transformed_matrix in enumerate(transformed_matrices):
            for vdx, training_matrix in enumerate(training_matrices):
                distance = self.distance_function(transformed_matrix, training_matrix)
                distance_matrix[udx][vdx] = distance

        results = [[vector.argmax(), vector.max()] if self.is_max else [vector.argmin(), vector.min()] for vector in
                   distance_matrix]

        return results

    def _calculate_projections(self, X: np.ndarray, Y: np.ndarray, rows: bool = True,
                               with_RRPP: bool = False) -> tuple[np.ndarray, np.ndarray, bool]:

        C_xx = self._calculate_cov(X, X, rows=rows)
        C_yy = self._calculate_cov(Y, Y, rows=rows)
        C_xy = self._calculate_cov(X, Y, rows=rows)
        C_yx = self._calculate_cov(Y, X, rows=rows)

        C_xx = C_xx + self.regular_C * np.identity(C_xx.shape[0])
        C_yy = C_yy + self.regular_C * np.identity(C_yy.shape[0])

        S_x = np.linalg.inv(C_xx) @ C_xy @ np.linalg.inv(C_yy) @ C_yx
        S_y = np.linalg.inv(C_yy) @ C_yx @ np.linalg.inv(C_xx) @ C_xy

        S_x = S_x + self.regular_S * np.identity(S_x.shape[0])
        S_y = S_y + self.regular_S * np.identity(S_y.shape[0])

        l_x, v_x = np.linalg.eig(S_x)
        l_y, v_y = np.linalg.eig(S_y)

        if with_RRPP:
            v_x = v_x.T[:, l_x.argsort()[-self.dimension:][::-1]]
            v_y = v_y.T[:, l_y.argsort()[-self.dimension:][::-1]]
        else:
            v_x = v_x[:, l_x.argsort()[::-1]].T
            v_y = v_y[:, l_y.argsort()[::-1]].T

        return v_x, v_y

    def _calculate_cov(self, matrixes_1: np.ndarray, matrixes_2: np.ndarray,
                       rows: bool = True) -> np.ndarray:

        if rows:
            C = np.sum([matrixes_1[i] @ matrixes_2[i].T for i in range(len(matrixes_1))], axis=0)
        else:
            C = np.sum([matrixes_1[i].T @ matrixes_2[i] for i in range(len(matrixes_1))], axis=0)
        return C


class TDCCAP(ClassifierAlgorithm):
    def __init__(self, dimension: int = 10, distance_function=lambda x, y: np.linalg.norm(x - y),
                 is_max: bool = True) -> None:

        self.U = None
        self.V = None
        self.W = {"x": [None, None], "y": [None, None]}
        self.mean_X = None
        self.mean_Y = None
        self.links_x = None
        self.links_y = None
        self.dimension = dimension
        self.distance_function = distance_function
        self.regular_C: float = 10 ** (-4)
        self.regular_S: float = 5 * 10 ** (-4)
        self.is_max = is_max

    def fit(self, X: list, Y: list, with_RRPP: bool = False) -> None:
        self.mean_X = np.mean(X, axis=0)
        self.mean_Y = np.mean(Y, axis=0)
        X -= self.mean_X
        Y -= self.mean_Y
        self.X = X
        self.Y = Y

        self._calculate_weights(X, Y, with_RRPP)

        self.U = self.transform(X + self.mean_X, is_x=True)

        self.V = self.transform(Y + self.mean_Y, is_x=False)

    def _calculate_weights(self, X, Y, with_RRPP):
        q1 = Queue()
        q2 = Queue()
        thread1 = Thread(target=self._worker, args=(q1, self._calculate_projections, X, Y, True, with_RRPP))
        thread2 = Thread(target=self._worker, args=(q2, self._calculate_projections, X, Y, False, with_RRPP))
        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        result1 = q1.get()
        result2 = q2.get()
        self.W["x"][0] = result1[0]
        self.W["y"][0] = result1[1]
        self.W["x"][1] = result2[0]
        self.W["y"][1] = result2[1]

    def _worker(self, queue, func, *args, **kwargs):
        queue.put(func(*args, **kwargs))

    def transform(self, matrix_set: np.ndarray, is_x: bool = True) -> np.ndarray:
        return np.array([self.W["x"][0].T @ (matrix - self.mean_X) @ self.W["x"][1] if is_x else self.W["y"][0].T @ (
                matrix - self.mean_X) @ self.W["y"][1] for matrix in matrix_set])

    def predict(self, matrices: list, is_x: bool = True) -> list[tuple[int, float]]:

        print(matrices)

        transformed_matrices = self.transform(matrices, is_x=is_x)
        training_matrices = self.U if is_x else self.V

        distance_matrix = np.zeros((len(matrices), len(training_matrices)), dtype="float64")

        for udx, transformed_matrix in enumerate(transformed_matrices):
            for vdx, training_matrix in enumerate(training_matrices):
                distance = self.distance_function(transformed_matrix, training_matrix)
                distance_matrix[udx][vdx] = distance
        results = [[vector.argmax(), vector.max()] if self.is_max else [vector.argmin(), vector.min()] for vector in
                   distance_matrix]

        return results

    def _calculate_projections(self, X: np.ndarray, Y: np.ndarray, rows: bool = True,
                               with_RRPP: bool = False) -> tuple[np.ndarray, np.ndarray, bool]:

        C_xx = self._calculate_cov(X, X, rows=rows)
        C_yy = self._calculate_cov(Y, Y, rows=rows)
        C_xy = self._calculate_cov(X, Y, rows=rows)
        C_yx = self._calculate_cov(Y, X, rows=rows)

        C_xx = C_xx + self.regular_C * np.identity(C_xx.shape[0])
        C_yy = C_yy + self.regular_C * np.identity(C_yy.shape[0])

        S_x = np.linalg.inv(C_xx) @ C_xy @ np.linalg.inv(C_yy) @ C_yx
        S_y = np.linalg.inv(C_yy) @ C_yx @ np.linalg.inv(C_xx) @ C_xy

        S_x = S_x + self.regular_S * np.identity(S_x.shape[0])
        S_y = S_y + self.regular_S * np.identity(S_y.shape[0])

        l_x, v_x = np.linalg.eig(S_x)
        l_y, v_y = np.linalg.eig(S_y)

        if with_RRPP:
            v_x = v_x.T[:, l_x.argsort()[-self.dimension:][::-1]]
            v_y = v_y.T[:, l_y.argsort()[-self.dimension:][::-1]]
        else:
            v_x = v_x[:, l_x.argsort()[::-1]].T
            v_y = v_y[:, l_y.argsort()[::-1]].T

        return v_x, v_y, rows

    def _calculate_cov(self, matrixes_1: np.ndarray, matrixes_2: np.ndarray,
                       rows: bool = True) -> np.ndarray:

        if rows:
            C = np.sum([matrixes_1[i] @ matrixes_2[i].T for i in range(len(matrixes_1))], axis=0)
        else:
            C = np.sum([matrixes_1[i].T @ matrixes_2[i] for i in range(len(matrixes_1))], axis=0)
        return C
