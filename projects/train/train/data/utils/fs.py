import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import Optional, Tuple, Union

import s3fs
from botocore.exceptions import ClientError, ResponseStreamingError
from filelock import FileLock
from fsspec.exceptions import FSTimeoutError

# s3 retry configuration
retry_config = {"retries": {"total_max_attempts": 10, "mode": "adaptive"}}


def split_data_dir(data_dir: Union[str, Path]) -> Tuple[Optional[str], str]:
    """
    Check if a data directory specifies a remote s3
    source by including  `"s3://"` at the start of
    the directory name.
    """
    data_dir = str(data_dir)
    if data_dir.startswith("s3://"):
        bucket = data_dir.replace("s3://", "")

        # check if specified a target location to map to
        # by adding a colon at the end of our bucket
        bucket, *data_dir = bucket.split(":")
        data_dir = data_dir[0] if data_dir else None
        return bucket, data_dir
    else:
        return None, data_dir


def get_data_dir(data_dir: str) -> Path:
    # generate our local node data directory
    # if our specified data source is remote
    bucket, data_dir = split_data_dir(data_dir)
    if bucket is not None and data_dir is None:
        # we have remote data, but we didn't explicitly
        # specify a directory to download it to, so create
        # a tmp directory using the worker id so that each
        # worker process downloads its own copy of the data
        # only on its first training run
        tmpdir = gettempdir()
        logging.info("Downloading data to local tmp directory")
        data_dir = f"{tmpdir}/data-tmp"

    logging.info(f"Downloading data to {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    return Path(data_dir)


def _download(
    s3: s3fs.S3FileSystem, source: str, target: str, num_retries: int = 5
):
    """
    Cheap wrapper around s3.get to try to avoid issues
    from interrupted reads.
    """

    lockfile = target + ".lock"
    logging.info(f"Downloading {source} to {target}")
    for i in range(num_retries):
        with FileLock(lockfile):
            if os.path.exists(target):
                logging.info(
                    f"Object {source} already downloaded by another process"
                )
                return
            try:
                s3.get(source, target)
                break
            except (ResponseStreamingError, FSTimeoutError, ClientError):
                logging.info(
                    "Download attempt {} for object {} "
                    "was interrupted, retrying".format(i + 1, source)
                )
                time.sleep(5)
                try:
                    os.remove(target)
                except FileNotFoundError:
                    continue

    else:
        raise RuntimeError(
            "Failed to download object {} due to repeated "
            "connection interruptions".format(source)
        )


def download_training_data(bucket: str, data_dir: str):
    """
    Download s3 data if it doesn't exist.
    """
    data_dir = str(data_dir)
    logging.info(
        "Downloading data from S3 bucket {} to "
        "local directory {}".format(bucket, data_dir)
    )

    # make a local directory to cache data if it
    # doesn't already exist
    background_dir = f"{data_dir}/train/background"
    os.makedirs(background_dir, exist_ok=True)

    # check to make sure the specified bucket
    # actually has data to download
    s3 = s3fs.S3FileSystem(
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        config_kwargs=retry_config,
    )
    sources = s3.glob(f"{bucket}/train/background/*.hdf5")
    if not sources:
        raise ValueError(f"No data at {bucket} to download")

    # multiprocess download of training data
    targets = [data_dir + f.replace(f"{bucket}", "") for f in sources]

    download = partial(_download, s3)
    with ThreadPoolExecutor() as executor:
        executor.map(download, sources, targets)
