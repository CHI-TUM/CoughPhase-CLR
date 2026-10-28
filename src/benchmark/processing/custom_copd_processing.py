import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
import os

data_dir = "datasets/coughCOPD/"
feature_dir = "feature/custom_copd_eval/"
audio_dir = data_dir + "default"


def extract_and_save_eGeMAPS():
    from src.benchmark.baseline.extract_feature import (
        extract_opensmile_features,
    )

    opensmile_features = []
    filenames = os.listdir(audio_dir)

    for filename in tqdm(filenames):
        file_path = os.path.join(audio_dir, filename)

        opensmile_feature = extract_opensmile_features(file_path)
        opensmile_features.append(opensmile_feature)
    np.save(
        feature_dir + "opensmile_feature",
        np.array(opensmile_features),
    )


def extract_and_save_embeddings(
    feature="operaCE", input_sec=2, dim=1280, output_file_name="operaCE"
):
    from src.benchmark.model_util import extract_opera_feature

    metadata_df = pd.read_csv(feature_dir + "meta_data.csv")

    filenames_np = (audio_dir + "/" + metadata_df["filename"].astype(str)).to_numpy()

    opera_features = extract_opera_feature(
        filenames_np, pretrain=feature, input_sec=input_sec, dim=dim
    )
    feature += str(dim)
    np.save(
        feature_dir + output_file_name + "_feature.npy",
        np.array(opera_features),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain",
        type=str,
    )
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--min_len_cnn", type=int, default=2)
    parser.add_argument("--min_len_htsat", type=int, default=2)
    parser.add_argument("--output_file_name", type=str, default="")

    args = parser.parse_args()

    if args.pretrain == "operaCT":
        input_sec = args.min_len_htsat
    elif args.pretrain == "operaCE":
        input_sec = args.min_len_cnn
    elif args.pretrain == "operaGT":
        input_sec = 8.18
    else:
        input_sec = args.min_len_cnn
    if args.pretrain == "opensmile":
        extract_and_save_eGeMAPS()
    else:
        extract_and_save_embeddings(
            args.pretrain,
            input_sec,
            dim=args.dim,
            output_file_name=args.output_file_name,
        )
