"""
The base/pretraining dataset is a set of Arrow files (HF dataset shards).
This file contains utilities for:
- iterating over the shards and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
# Korean-only FineWeb2 subset hosted as Arrow shards
DATASET_ID = "minpeter/fineweb-2-edu-korean-raw"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"

# Manifest of shards (from HF repo listing). 502 train shards + 1 small eval shard.
TRAIN_SHARDS = [f"data-{i:05d}-of-00502.arrow" for i in range(502)]
VAL_SHARDS = ["data-00000-of-00001.arrow"]  # small held-out shard
ALL_SHARDS = TRAIN_SHARDS + VAL_SHARDS
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Helpers to map shard names to local paths and URLs
def _shard_relpath(shard):
    # Arrow shards are already flat filenames
    return shard

def shard_to_local_path(shard, data_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir
    relpath = _shard_relpath(shard)
    local_path = os.path.join(data_dir, relpath)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return local_path

def shard_to_url(shard):
    # Use ?download=1 to fetch the actual LFS binary, not the pointer
    return f"{BASE_URL}/{shard}?download=1"

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, split=None, strict=False):
    """
    Return local paths to all shards for a split.
    - split can be "train", "val", or None (all).
    - if strict=True, raises if any requested shard is missing.
    """
    data_dir = DATA_DIR if data_dir is None else data_dir
    if split is None:
        shards = ALL_SHARDS
    elif split == "train":
        shards = TRAIN_SHARDS
    elif split == "val":
        shards = VAL_SHARDS
    else:
        raise ValueError("split must be None, 'train', or 'val'")

    missing = []
    paths = []
    for shard in shards:
        local_path = shard_to_local_path(shard, data_dir)
        if os.path.exists(local_path):
            paths.append(local_path)
        else:
            missing.append(local_path)

    if not paths:
        raise FileNotFoundError(
            f"No shards found for split={split or 'all'}; "
            f"expected examples: {shards[:3]}"
        )

    if strict and missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} shard(s) for split={split or 'all'}; "
            f"first missing: {missing[:3]}"
        )
    if missing:
        print(f"Warning: skipping {len(missing)} missing shard(s) for split={split or 'all'}")
    return paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val".
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(split=split)
    import pyarrow.ipc as ipc

    def arrow_batches(path):
        # Try file reader first, fall back to stream reader
        try:
            reader = ipc.open_file(path)
            num_batches = reader.num_record_batches
            for batch_idx in range(start, num_batches, step):
                yield reader.get_batch(batch_idx)
        except Exception:
            reader = ipc.open_stream(path)
            for idx, batch in enumerate(reader):
                if idx < start:
                    continue
                if (idx - start) % step == 0:
                    yield batch

    for filepath in parquet_paths:
        if filepath.endswith(".parquet"):
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(start, pf.num_row_groups, step):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                yield texts
        else:
            for batch in arrow_batches(filepath):
                texts = batch.column('text').to_pylist()
                yield texts

# -----------------------------------------------------------------------------
def download_single_file(shard):
    """ Downloads a single shard path, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filepath = shard_to_local_path(shard)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = shard_to_url(shard)
    print(f"Downloading {shard}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {shard}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {shard}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {shard} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb2 Korean dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    total_train = len(TRAIN_SHARDS)
    num_train = total_train if args.num_files == -1 else min(args.num_files, total_train)

    # Always download the val shard so eval scripts work
    shards_to_download = TRAIN_SHARDS[:num_train] + VAL_SHARDS

    print(f"Downloading {len(shards_to_download)} shards ({num_train} train + {len(VAL_SHARDS)} val) using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, shards_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(shards_to_download)} shards to {DATA_DIR}")
