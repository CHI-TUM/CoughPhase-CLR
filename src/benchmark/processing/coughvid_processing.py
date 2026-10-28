import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
import os

data_dir = "datasets/coughvid/"
feature_dir = "feature/coughvid_eval/"
audio_dir = data_dir + "wav"
if not os.path.exists(audio_dir):
    print(f"Folder not found: {audio_dir}, downloading the dataset")
    os.system("sh datasets/coughvid/download_data.sh")
    # raise FileNotFoundError(
    #     f"Folder not found: {audio_dir}, please download the dataset.")


train_uuid = np.load(data_dir + "coughvid__train_uuids.npy", allow_pickle=True)
val_uuid = np.load(data_dir + "coughvid__val_uuids.npy", allow_pickle=True)
covid_test_uuid = np.load(data_dir + "coughvid_covid_test_uuids.npy", allow_pickle=True)
gender_test_uuid = np.load(
    data_dir + "coughvid_gender_test_uuids.npy", allow_pickle=True
)
all_uuid = list(train_uuid) + list(val_uuid) + list(gender_test_uuid)


def preprocess_label(label="covid"):
    df = pd.read_csv(data_dir + "metadata_compiled.csv", index_col="uuid")
    df = df.replace(np.nan, "", regex=True)
    # df = df[df["gender"].str.contains("male")]

    gender_label_dict = {
        "female": 1,
        "male": 0,
        "pnts": None,
        "Other": None,
        "other": None,
        "": None,
    }
    covid_label_dict = {
        "COVID-19": 1,
        "healthy": 0,
        "pnts": None,
        "Other": None,
        "symptomatic": None,
        "": None,
    }

    filename_list = []
    label_list = []
    split = []
    for uuid, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = data_dir + "wav/" + uuid + ".wav"
        if not exists(filename):
            # problem in data name
            filename = data_dir + "wav/" + uuid[:-1] + ".wav"
        if label == "gender":
            audio_label = gender_label_dict[row["gender"]]

        elif label == "covid":
            audio_label = covid_label_dict[row["status"]]

        if audio_label is None:
            continue
        if uuid not in all_uuid:
            # no in downstream
            continue

        label_list.append(audio_label)
        filename_list.append(filename)
        if uuid in train_uuid:
            split.append("train")
        elif uuid in val_uuid:
            split.append("val")
        else:
            split.append("test")

    np.save(feature_dir + "label_{}.npy".format(label), label_list)
    np.save(feature_dir + "sound_dir_loc_{}.npy".format(label), filename_list)
    np.save(feature_dir + "split_{}.npy".format(label), split)


def extract_and_save_embeddings(
    feature="operaCE", label="covid", input_sec=2, dim=1280, output_file_name="operaCE"
):
    from src.benchmark.model_util import extract_opera_feature

    sound_dir_loc = np.load(feature_dir + "sound_dir_loc_{}.npy".format(label))
    opera_features = extract_opera_feature(
        sound_dir_loc, pretrain=feature, input_sec=input_sec, dim=dim
    )
    feature += str(dim)
    np.save(
        feature_dir + output_file_name + "_feature_{}.npy".format(label),
        np.array(opera_features),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain",
        type=str,
        default="cks/model/combined/coughvid_covidUKcough/encoder-crop_20000_steps_data_used_20-epoch=849--valid_acc=0.04-valid_loss=4.6404.ckpt",
    )
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--label", type=str, default="gender")
    parser.add_argument("--min_len_cnn", type=int, default=2)
    parser.add_argument("--min_len_htsat", type=int, default=2)
    parser.add_argument("--output_file_name", type=str, default="")
    parser.add_argument("--input_sec", type=int, default=None)

    args = parser.parse_args()

    if args.pretrain == "operaCT":
        input_sec = args.min_len_htsat
    elif args.pretrain == "operaCE":
        input_sec = args.min_len_cnn
    elif args.pretrain == "operaGT":
        input_sec = 8.18
    else:
        input_sec = args.min_len_cnn

    output_file_name = args.output_file_name
    if args.input_sec is not None and args.input_sec != input_sec:
        assert output_file_name, "--input_sec needs --output_file_name"
        output_file_name += "_sec{}".format(args.input_sec)
        input_sec = args.input_sec

    extract_and_save_embeddings(
        args.pretrain,
        args.label,
        input_sec,
        dim=args.dim,
        output_file_name=output_file_name,
    )
