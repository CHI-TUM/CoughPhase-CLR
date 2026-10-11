import argparse
import numpy as np
from tqdm import tqdm
from src.util import get_entire_signal_librosa
import os
from glob import glob

# for pretraining


def load_test_uuids():
    # union of the benchmark test splits (covid test is a subset of gender test)
    test_uuids = set()
    for split in ["gender", "covid"]:
        test_uuids.update(
            np.load(
                f"datasets/coughvid/coughvid_{split}_test_uuids.npy", allow_pickle=True
            ).tolist()
        )
    # some wav files are named uuid[:-1] (see coughvid_processing.py)
    return test_uuids | {uuid[:-1] for uuid in test_uuids}


def is_test_recording(segment_name, test_uuids):
    # segments are named <recording>_cough_<i>
    return segment_name.rsplit("_cough_", 1)[0] in test_uuids


def preprocess_spectrogram_SSL(
    modality="modality", input_sec=2, form="", nfft=1024, hop=512, TRIM=True
):

    path = "datasets/coughvid/"

    output_dir = f"datasets/coughvid/entire_spec_npy{form}/"
    os.makedirs(output_dir, exist_ok=True)

    filenames = glob(path + f"wav{form}/*.wav")

    filenames = [f"{os.path.basename(f)}" for f in filenames]
    # keep benchmark test recordings out of pretraining
    test_uuids = load_test_uuids()
    filenames = [
        f for f in filenames if not is_test_recording(f.split(".")[0], test_uuids)
    ]
    invalid_data = 0

    filename_list = []

    for file in tqdm(filenames):

        userID = file.split(".")[0]

        if os.path.exists(path + f"wav{form}/" + file):
            data = get_entire_signal_librosa(
                f"datasets/coughvid/wav{form}",
                userID,
                spectrogram=True,
                input_sec=input_sec,
                nfft=nfft,
                hop=hop,
                TRIM=TRIM,
            )

            if data is None:
                invalid_data += 1
                continue

            np.save(output_dir + userID + ".npy", data)
            filename_list.append(f"datasets/coughvid/entire_spec_npy{form}/" + userID)

    np.save(f"datasets/coughvid/entire_spec_filenames{form}.npy", filename_list)
    print(
        "finished preprocessing cough: valid data",
        len(filename_list),
        "; invalid data",
        invalid_data,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="cough")
    parser.add_argument("--input_sec", type=int, default=0)
    parser.add_argument("--form", type=str, default="_segmented_phase_2")
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--TRIM", type=bool, default=True)

    args = parser.parse_args()

    preprocess_spectrogram_SSL(
        modality=args.modality,
        input_sec=args.input_sec,
        form=args.form,
        nfft=args.nfft,
        hop=args.hop,
        TRIM=args.TRIM,
    )
