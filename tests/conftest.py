"""
Shared test fixtures and utilities for AMPLFI tests.
"""

import numpy as np
import pytest
import torch.nn as nn
from gwpy.timeseries import TimeSeries
from pathlib import Path

from amplfi.train.architectures.embeddings.base import Embedding


class SimpleTestEmbedding(Embedding):
    def __init__(self, num_ifos: int, context_dim: int = 4):
        super().__init__()
        self.num_ifos = num_ifos
        self.context_dim = context_dim

        self.linear = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        self.final_linear = nn.Linear(num_ifos * 8, context_dim)

    def forward(self, context):
        import torch

        strain, _ = context

        batch_size, num_ifos, strain_length = strain.shape
        x = strain.reshape(-1, 1)

        x = self.linear(x)
        x = self.relu(x)

        x = x.reshape(batch_size, num_ifos, strain_length, 8)
        x = torch.mean(x, dim=2)

        x = x.reshape(batch_size, num_ifos * 8)
        x = self.final_linear(x)

        return x


@pytest.fixture
def create_mock_data():
    """
    Fixture that returns a function to create mock HDF5 strain data files
    using gwpy.TimeSeries for realistic LIGO data format.
    """

    def _create_mock_data_files(
        data_dir: Path, sample_rate: int = 512, duration_per_file: int = 120
    ):
        """
        Create mock HDF5 strain data files using gwpy.TimeSeries.

        Args:
            data_dir: Directory to create data files in
            sample_rate: Sample rate in Hz (default: 512)
            duration_per_file: Duration of each file in seconds (default: 120)
        """

        # Create training data structure: data_dir/train/background/*.hdf5
        train_bg_dir = data_dir / "train" / "background"
        train_bg_dir.mkdir(parents=True)

        # Create test data structure: data_dir/test/background/*.hdf5
        test_bg_dir = data_dir / "test" / "background"
        test_bg_dir.mkdir(parents=True)

        # Create mock strain data files
        # File naming convention: {prefix}-{gps_start}-{duration}.hdf5
        gps_start = 1000000000  # Some GPS time

        for split, bg_dir in [("train", train_bg_dir), ("test", test_bg_dir)]:
            # Create 2 files for training, 1 for testing
            n_files = 2 if split == "train" else 1

            for i in range(n_files):
                file_gps = gps_start + i * duration_per_file
                filename = f"background-{file_gps}-{duration_per_file}.hdf5"
                filepath = bg_dir / filename

                # Create TimeSeries for each IFO and write to HDF5
                for ifo in ["H1", "L1"]:
                    # Create realistic noise-like strain data
                    strain_data = np.random.normal(
                        0, 1e-21, sample_rate * duration_per_file
                    )

                    # Create TimeSeries object with proper metadata
                    ts = TimeSeries(
                        strain_data,
                        t0=file_gps,
                        dt=1 / sample_rate,
                        name=ifo,
                    )

                    # Write to HDF5 - this will create all the proper metadata
                    ts.write(filepath, path=ifo, append=True)

    return _create_mock_data_files
