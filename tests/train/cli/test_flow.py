import os
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestAmplfiFlowCLI:
    """Integration tests for amplfi-flow-cli command."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for data and output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "data"
            out_dir = temp_path / "output"
            data_dir.mkdir()
            out_dir.mkdir()

            # Set required environment variables
            os.environ["AMPLFI_DATADIR"] = str(data_dir)
            os.environ["AMPLFI_OUTDIR"] = str(out_dir)

            yield data_dir, out_dir

            # Clean up environment variables
            if "AMPLFI_DATADIR" in os.environ:
                del os.environ["AMPLFI_DATADIR"]
            if "AMPLFI_OUTDIR" in os.environ:
                del os.environ["AMPLFI_OUTDIR"]

    def test_flow_training_end_to_end(self, temp_dirs, create_mock_data):
        """Test that amplfi-flow-cli runs end-to-end without errors."""
        data_dir, out_dir = temp_dirs

        # Get path to our test config
        config_path = Path(__file__).parent / "config.yaml"

        # Create mock strain data files
        create_mock_data(data_dir)

        # Run the CLI command
        cmd = ["amplfi-flow-cli", "fit", f"--config={config_path}"]

        # Execute the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for safety
        )

        assert result.returncode == 0, (
            f"CLI failed with stderr: {result.stderr}"
        )
        assert "max_epochs=2` reached" in result.stderr, (
            f"Training did not complete properly. stderr: {result.stderr}"
        )
