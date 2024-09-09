from numpy import array, atleast_2d, ndarray, diag, sqrt, outer
from scipy.sparse import isspmatrix

class KinterfaceCombine:
    """
    Combine multiple Kinterface kernels using the given weights.
    """

    def __init__(self, kernels, weights):
        """
        :param kernels: (list) List of Kinterface objects.
        :param weights: (list) List of weights for each kernel.
        """
        assert len(kernels) == len(weights), "Number of kernels and weights must match."
        self.kernels = kernels
        self.weights = weights
        self.data = kernels[0].data
        self.kernel_args = kernels[0].kernel_args
        self.data_labels = kernels[0].data_labels
        self.row_normalize = kernels[0].row_normalize
        self.shape = kernels[0].shape

    def __getitem__(self, item):
        """
        Access portions of the combined kernel matrix.

        :param item: (tuple) pair of: indices or list of indices or (numpy.ndarray) or (slice) to address portions of the kernel matrix.
        :return: (numpy.ndarray) Value of the combined kernel matrix for item.
        """
        assert isinstance(item, tuple)
        assert len(item) == 2
        combined_value = sum(weight * kernel[item] for kernel, weight in zip(self.kernels, self.weights))
        return combined_value

    def __call__(self, x, y):
        """
        Mimic a callable kernel function.

        :param x: (numpy.ndarray) Data point.
        :param y: (numpy.ndarray) Data point.
        :return: (numpy.ndarray) Value of the combined kernel for x, y.
        """
        combined_value = sum(weight * kernel(x, y) for kernel, weight in zip(self.kernels, self.weights))
        return combined_value

    def diag(self):
        """
        Diagonal of the combined kernel matrix.

        :return (numpy.ndarray) diagonal values of the combined kernel matrix.
        """
        return array([self[i, i] for i in range(self.shape[0])]).ravel()

    def diagonal(self):
        """ Diagonal of the combined kernel matrix (alias). """
        return self.diag()