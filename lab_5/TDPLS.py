import numpy as np
from threading import Thread
from lab_5.modules.classifier import ClassifierAlgorithm
from queue import Queue

class TDPLSC(ClassifierAlgorithm):
    def __init__(self, dimension: int = 10, distance_function=lambda x, y: np.linalg.norm(x - y),
                 is_max: bool = True) -> None:

        self.U = None
        self.V = None
        self.W = {"x": [None, None], "y": [None, None]}
        self.X_c = None
        self.Y_c = None
        self.links_x = None
        self.links_y = None
        self.dimension = dimension
        self.distance_function = distance_function
        self.regular_s: float = 5 * 10 ** (-4)
        self.is_max = is_max

    def fit(self, X: list, Y: list, with_RRPP: bool = False) -> None:
        self.X_c = np.mean(X, axis=0)
        self.Y_c = np.mean(Y, axis=0)

        X -= self.X_c
        Y -= self.Y_c

        self.W["x"][0], self.W["y"][0] = self._calculate_projections(X, Y, rows=True, with_RRPP=with_RRPP)

        self.U = [self.W['x'][0].T @ x for x in X]
        self.V = [self.W["y"][0].T @ y for y in Y]

        self.W["x"][1], self.W["y"][1] = self._calculate_projections(X, Y, rows=False, with_RRPP=with_RRPP)

        self.U = np.array([u @ self.W["x"][1] for u in self.U])
        self.V = np.array([v @ self.W["y"][1] for v in self.V])

    def transform(self, matrix_set: np.ndarray, is_x: bool = True) -> np.ndarray:
        W = self.W["x"] if is_x else self.W["y"]
        X_c = self.X_c if is_x else self.Y_c
        return np.array([W[0].T @ (m - X_c) @ W[1] for m in matrix_set])

    def predict(self, matrixes: list, is_x: bool = True) -> tuple[int, float]:

        transformed_matrices = self.transform(matrixes, is_x=is_x)
        training_matrices = self.U if is_x else self.V

        distance_matrix = np.zeros((len(matrixes), len(training_matrices)), dtype="float64")

        for udx, transformed_matrix in enumerate(transformed_matrices):
            for vdx, training_matrix in enumerate(training_matrices):
                distance = self.distance_function(transformed_matrix, training_matrix)
                distance_matrix[udx][vdx] = distance

        results = [[vector.argmax(), vector.max()] if self.is_max else [vector.argmin(), vector.min()] for vector in
                   distance_matrix]

        return results

    def _calculate_projections(self, X: np.ndarray, Y: np.ndarray, rows: bool = True,
                               with_RRPP: bool = False) -> tuple[np.ndarray, np.ndarray]:
        C_xy = self._calculate_covariance_matrix(X, Y, rows=rows)
        C_yx = self._calculate_covariance_matrix(Y, X, rows=rows)

        S_x = C_xy @ C_yx
        S_y = C_yx @ C_xy

        S_x = S_x + self.regular_s * np.identity(S_x.shape[0])
        S_y = S_y + self.regular_s * np.identity(S_y.shape[0])

        l_x, v_x = np.linalg.eig(S_x)
        l_y, v_y = np.linalg.eig(S_y)

        if with_RRPP:
            v_x = v_x.T[:, l_x.argsort()[-self.dimension:][::-1]]
            v_y = v_y.T[:, l_y.argsort()[-self.dimension:][::-1]]
        else:
            v_x = v_x[:, l_x.argsort()[::-1]].T
            v_y = v_y[:, l_y.argsort()[::-1]].T

        return v_x, v_y

    def _calculate_covariance_matrix(self, matrixes_1: np.ndarray, matrixes_2: np.ndarray,
                                     rows: bool = True) -> np.ndarray:
        C = np.sum([matrixes_1[i] @ matrixes_2[i].T if rows else matrixes_1[i].T @ matrixes_2[i] for i in
                    range(len(matrixes_1))], axis=0)
        return C


class TDPLSP(ClassifierAlgorithm):
    def __init__(self, dimension: int = 10, distance_function=lambda x, y: np.linalg.norm(x - y),
                 is_max: bool = True) -> None:

        self.U = None
        self.V = None
        self.W = {"x": [None, None], "y": [None, None]}
        self.X_c = None
        self.Y_c = None
        self.links_x = None
        self.links_y = None
        self.dimension = dimension
        self.distance_function = distance_function
        self.regular_s: float = 5 * 10 ** (-4)
        self.is_max = is_max

    def fit(self, X: list, Y: list, with_RRPP: bool = False) -> None:

        self.X_c = np.mean(X, axis=0)
        self.Y_c = np.mean(Y, axis=0)

        X -= self.X_c
        Y -= self.Y_c

        self._calculate_weights(X, Y, with_RRPP)

        self.U = self.transform(X + self.X_c, is_x=True)

        self.V = self.transform(Y + self.X_c, is_x=False)

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
        W = self.W["x"] if is_x else self.W["y"]
        X_c = self.X_c if is_x else self.Y_c
        return np.array([W[0].T @ (m - X_c) @ W[1] for m in matrix_set])

    def predict(self, matrixes: list, is_x: bool = True) -> tuple[int, float]:

        transformed_matrices = self.transform(matrixes, is_x=is_x)
        training_matrices = self.U if is_x else self.V

        distance_matrix = np.zeros((len(matrixes), len(training_matrices)), dtype="float64")

        for udx, transformed_matrix in enumerate(transformed_matrices):
            for vdx, training_matrix in enumerate(training_matrices):
                distance = self.distance_function(transformed_matrix, training_matrix)
                distance_matrix[udx][vdx] = distance

        results = [[vector.argmax(), vector.max()] if self.is_max else [vector.argmin(), vector.min()] for vector in
                   distance_matrix]

        return results

    def _calculate_projections(self, X: np.ndarray, Y: np.ndarray, rows: bool = True,
                               with_RRPP: bool = False) -> tuple[np.ndarray, np.ndarray]:
        C_xy = self._calculate_covariance_matrix(X, Y, rows=rows)
        C_yx = self._calculate_covariance_matrix(Y, X, rows=rows)

        S_x = C_xy @ C_yx
        S_y = C_yx @ C_xy

        S_x = S_x + self.regular_s * np.identity(S_x.shape[0])
        S_y = S_y + self.regular_s * np.identity(S_y.shape[0])

        l_x, v_x = np.linalg.eig(S_x)
        l_y, v_y = np.linalg.eig(S_y)

        if with_RRPP:
            v_x = v_x.T[:, l_x.argsort()[-self.dimension:][::-1]]
            v_y = v_y.T[:, l_y.argsort()[-self.dimension:][::-1]]
        else:
            v_x = v_x[:, l_x.argsort()[::-1]].T
            v_y = v_y[:, l_y.argsort()[::-1]].T

        return v_x, v_y

    def _calculate_covariance_matrix(self, matrixes_1: np.ndarray, matrixes_2: np.ndarray,
                                     rows: bool = True) -> np.ndarray:
        C = np.sum([matrixes_1[i] @ matrixes_2[i].T if rows else matrixes_1[i].T @ matrixes_2[i] for i in
                    range(len(matrixes_1))], axis=0)
        return C