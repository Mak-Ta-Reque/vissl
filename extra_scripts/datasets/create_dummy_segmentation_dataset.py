# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bisect
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm



def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the folder containing the original CLEVR_v1.0 dataset",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Folder where the classification dataset will be written",
    )
    parser.add_argument(
        "-d",
        "--download",
        action="store_const",
        const=True,
        default=False,
        help="To download the original dataset and decompress it in the input folder",
    )
    return parser


def download_dataset(root: str):
    """
    Download the CLEVR dataset archive and expand it in the folder provided as parameter
    """
    URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
    download_and_extract_archive(url=URL, download_root=root)



if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_clevr_dist_data_files.py -i /path/to/clevr/ \
        -o /output_path/to/clevr_dist
    ```
    """
    args = get_argument_parser().parse_args()
    output_path = args.output
    input_dir = args.input
    image_path = os.path.join(input_dir, "images")
    mask_path = os.path.join(input_dir, "masks")
    images = [os.path.join(image_path, im) for im in os.listdir(mask_path) if ".png" in im]
    masks = [os.path.join(mask_path, im) for im in os.listdir(mask_path) if ".png" in im]
    images = np.save(os.path.join(output_path, "images.npy"), images, allow_pickle=True)
    masks = np.save(os.path.join(output_path, "masks.npy"), masks, allow_pickle=True)
    



