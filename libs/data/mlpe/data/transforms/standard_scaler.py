import torch
from mlpe.data.transforms.transform import Transform


class StandardScalerTransform(Transform):
    def __init__(self, num_parameters: int):

        super().__init__()
        self.mean = self.add_parameter(torch.zeros([num_parameters]))
        self.std = self.add_parameter(torch.ones([num_parameters]))

    def fit(self, X: torch.Tensor) -> None:
        if X.ndim != 2:
            raise ValueError(
                "Can only fit StandardScaler on 2 dimensional input"
            )

        # take mean along "columns" of data
        # where each column is a different
        # parameter
        self.set_value(self.mean, X.mean(axis=0))
        self.set_value(self.std, X.std(axis=0))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 2 or X.shape[1] != len(self.mean):
            raise ValueError(
                "Can't transform tensor of shape {}"
                "using StandardScaler for {} parameters".format(
                    X.shape, len(self.mean)
                )
            )

        X = (X - self.mean) / self.std
        return X
