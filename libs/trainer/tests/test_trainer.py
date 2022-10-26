import pytest
import torch
from mlpe.trainer.trainer import train_for_one_epoch
from nflows import distributions, flows, transforms


@pytest.fixture(params=[1])
def n_features(request):
    return request.param


def make_bimodal_data(n_features):
    dist_1 = torch.randn(size=(4000, n_features)) - 5
    dist_2 = torch.randn(size=(4000, n_features)) + 5

    context_1 = torch.zeros_like(dist_1)
    context_2 = torch.ones_like(dist_2)

    data = torch.concat([dist_1, dist_2])
    context = torch.concat([context_1, context_2])
    dataloader = Dataloader(data, context, 32)

    return dataloader


class Dataloader:
    def __init__(self, data, context, batch_size):
        self.data = data
        self.context = context
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.data) - 1) // self.batch_size + 1

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if (self.i + 1) == len(self):
            raise StopIteration

        slc = slice(self.i * self.batch_size, (self.i + 1) * self.batch_size)
        self.i += 1
        return self.context[slc], self.data[slc]


def make_simple_flow(n_features):
    num_layers = 5
    base_dist = distributions.StandardNormal(
        shape=[n_features],
    )

    transform_list = []
    for _ in range(num_layers):
        transform_list.append(
            transforms.ReversePermutation(features=n_features)
        )
        transform_list.append(
            transforms.MaskedAffineAutoregressiveTransform(
                features=n_features, hidden_features=4, context_features=1
            )
        )
    transform = transforms.CompositeTransform(transform_list)

    flow = flows.Flow(transform, base_dist)
    return flow


def test_train_one_epoch(n_features):

    train_dataset = make_bimodal_data(n_features)
    valid_dataset = make_bimodal_data(n_features)

    flow = make_simple_flow(n_features)

    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    for i in range(20):
        train_loss, valid_loss, duration, throughput = train_for_one_epoch(
            flow,
            optimizer,
            train_dataset,
            valid_dataset,
        )
        print(train_loss, valid_loss)
    samples = (
        torch.flatten(flow.sample(100, torch.Tensor([[0]]))).detach().numpy()
    )

    assert (samples < 0).all()

    samples = (
        torch.flatten(flow.sample(100, torch.Tensor([[1]]))).detach().numpy()
    )
    assert (samples > 0).all()
