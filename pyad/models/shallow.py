import numpy as np

from abc import ABC, abstractmethod
from pyad.utilities.cli import MODEL_REGISTRY
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class BaseShallowModel(ABC):
    def __init__(self, **kwargs):
        self.clf = None

    @abstractmethod
    def print_name(self):
        pass

    @abstractmethod
    def get_hparams(self):
        pass

    def fit(self, train_data: np.ndarray) -> None:
        self.clf.fit(train_data)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.score(X)

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        pass

    def get_params(self) -> dict:
        return self.get_hparams()


@MODEL_REGISTRY
class OCSVM(BaseShallowModel):
    def __init__(
            self,
            kernel="rbf",
            nu=0.5,
            gamma="scale",
            shrinking=False,
            verbose=True,
            **kwargs
    ):
        """
            kernel: str
                from sklearn: specifies the kernel type to be used in the algorithm
            gamma: str
                from sklearn: kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            nu: float
             from sklearn: an upper bound on the fraction of training errors and a lower bound of the fraction of
             support vectors (should be in the interval (0, 1])
        ]
        """
        super(OCSVM, self).__init__(**kwargs)
        self.gamma = gamma if not any(char.isdigit() for char in gamma) else float(gamma)
        self.kernel = kernel
        self.nu = nu
        self.shrinking = shrinking
        self.verbose = verbose
        self.clf = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            shrinking=shrinking,
            verbose=verbose,
            nu=nu
        )

    def print_name(self):
        return "OC-SVM"

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.clf.score_samples(X)

    def get_hparams(self) -> dict:
        return dict(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
            shrinking=self.shrinking
        )


@MODEL_REGISTRY
class LOF(BaseShallowModel):
    def __init__(
            self,
            n_neighbors: int,
            **kwargs
    ):
        """
        n_neighbors: int
            from sklearn: the actual number of neighbors used for :meth:`kneighbors` queries
        """
        super(LOF, self).__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            n_jobs=-1
        )

    def print_name(self):
        return "LOF"

    def get_hparams(self) -> dict:
        return dict(
            n_neighbors=self.n_neighbors,
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.clf.score_samples(X)
