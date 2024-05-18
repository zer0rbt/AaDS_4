import numpy as np

class Correlation:
    @staticmethod
    def distantion(X: np.ndarray, Y: np.ndarray) -> float:

        return np.sum(np.abs(X - Y))
    
    @staticmethod
    def average(matrix1: np.ndarray, matrix2: np.ndarray) -> float:

        return np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]

    @staticmethod
    def cov(matrix1: np.ndarray, matrix2: np.ndarray) -> float:

        cov_matrix = np.cov(matrix1.flatten(), matrix2.flatten())
        std_matrix = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.dot(std_matrix, std_matrix)
        return corr_matrix[0, 1]

    @staticmethod
    def pirson(matrix1: np.ndarray, matrix2: np.ndarray) -> float:

        vector1, vector2 = matrix1.flatten(), matrix2.flatten()
        covariance = np.cov(vector1, vector2)[0][1]
        return covariance / (np.std(vector1) * np.std(vector2))

    @staticmethod
    def cov_set(X: np.ndarray, Y: np.ndarray) -> float:

        return np.mean([Correlation.cov(x, y) for x, y in zip(X, Y)])

    @staticmethod
    def cov_element(matrix: np.ndarray) -> float:

        correlations = np.corrcoef(matrix, rowvar=False)
        num_pairs = matrix.shape[1] * (matrix.shape[1] - 1) / 2
        return np.sum(correlations) / num_pairs
