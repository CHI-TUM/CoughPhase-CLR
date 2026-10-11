# Yuwei (Evelyn) Zhang
# yz798@cam.ac.uk
# Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking
# https://github.com/evelyn0414/OPERA
# some code below is referenced from https://github.com/CVxTz/COLA_pytorch
from autrainer.augmentations import SpecAugment
import os
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from lightning.pytorch.utilities import CombinedLoader
import random
from src.util import random_crop, random_mask, random_multiply
from src.model.models_cola import Cola, ColaMD
import math


def combine_dataloaders(dataloaders, train=False):
    if train:
        return CombinedLoader(dataloaders, "max_size_cycle")
    return CombinedLoader(dataloaders, "sequential")


def mask_frequencies(x, side_to_mask):
    """
    Masks either the top half or bottom half of the frequency bins in the spectrogram.
    Args:
        x: Input spectrogram of shape (time, freq)
        side_to_mask: "top" to mask the top half, "bottom" to mask
                        the bottom half
    Returns:
        Masked spectrogram of shape (time, freq)
    """
    x = x.copy()

    num_freqs = x.shape[1]
    half_freq = num_freqs // 2

    if side_to_mask == "top":
        x[:, half_freq:] = 0
    elif side_to_mask == "bottom":
        x[:, :half_freq] = 0
    else:
        raise ValueError("side must be either 'left' or 'right'")
    return x


def pad(x):
    """
    Pads or truncates the spectrogram to a fixed length of 50 along the time dimension.
    Args:
        x: Input spectrogram of shape (time, freq)
    Returns:
        Padded or truncated spectrogram of shape (50, freq)
    """
    time_len = x.shape[0]
    if time_len < 50:
        padding_needed = 50 - time_len
        x_padded = np.pad(x, ((0, padding_needed), (0, 0)), mode="reflect")
    else:
        x_padded = x[:50]  # Truncate if longer than 25
    return x_padded


def interpolate_to_size(x, target_size=50):
    """
    Interpolates the spectrogram to the target size along the time dimension.

    Args:
        x: Input spectrogram of shape (time, freq)
        target_size: Target time dimension size

    Returns:
        Interpolated spectrogram of shape (target_size, freq)
    """
    import torch.nn.functional as F

    x = torch.from_numpy(x).float()

    x = x.unsqueeze(0).unsqueeze(0)

    interpolated = F.interpolate(
        x,
        size=(target_size, x.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )

    interpolated = interpolated.squeeze(0).squeeze(0)

    return interpolated.numpy()


output_dir = "spectrogram_images"
os.makedirs(output_dir, exist_ok=True)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        data_2=None,
        max_len=200,
        spec_augment=False,
        augment=True,
        from_npy=False,
        labels=None,
        method="cola",
        positive_pair_method="crop",
        preprocessing=None,
        data_percentage=1.0,
    ):
        """
        max len: 251 for 8 secs, 157 for 5 second, 126 for 4 seconds, 63 for 2 seconds, 32 for 1 second
        """
        self.data = data
        self.data_2 = data_2
        self.max_len = max_len
        self.augment = augment
        self.from_npy = from_npy
        self.labels = labels
        self.method = method
        self.positive_pair_method = positive_pair_method
        self.spec_augment = spec_augment
        self.preprocessing = preprocessing
        self.spec_augment_op = None
        if self.spec_augment:
            self.spec_augment_op = SpecAugment(W=10, time_mask=10, freq_mask=15)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.positive_pair_method == "phase":
            npy_path1 = self.data[idx]
            npy_path2 = self.data_2[idx]
            x1 = np.load(npy_path1 + ".npy")
            x2 = np.load(npy_path2 + ".npy")
            if os.path.basename(npy_path1) != os.path.basename(npy_path2):
                print("paths not equal: ", npy_path1, npy_path2)

        else:
            if self.from_npy:
                npy_path = self.data[idx]
                x = np.load(npy_path + ".npy")
            else:
                x = self.data[idx]

        if self.method == "cola":
            if self.augment:
                x = random_mask(x)

            if self.positive_pair_method == "phase":
                x1 = interpolate_to_size(x1)
                x2 = interpolate_to_size(x2)

            elif self.positive_pair_method == "crop":
                x1 = random_crop(x, crop_size=self.max_len)
                x2 = random_crop(x, crop_size=self.max_len)

            elif self.positive_pair_method == "mask":
                time_len = x.shape[0]
                if time_len > self.max_len:
                    start_idx = np.random.randint(0, time_len - self.max_len + 1)
                    x_cropped = x[start_idx : start_idx + self.max_len, :]
                elif time_len < self.max_len:
                    padding_needed = self.max_len - time_len
                    x_cropped = np.pad(
                        x, ((0, padding_needed), (0, 0)), mode="constant"
                    )
                else:
                    x_cropped = x

                x1 = mask_frequencies(x_cropped, side_to_mask="top")
                x2 = mask_frequencies(x_cropped, side_to_mask="bottom")
            else:
                raise ValueError("positive_pair_method must be either 'crop' or 'mask'")

            if self.augment:
                x1 = random_multiply(x1)
                x2 = random_multiply(x2)

            x1 = torch.tensor(x1, dtype=torch.float)
            x2 = torch.tensor(x2, dtype=torch.float)

            if self.spec_augment:

                x1 = x1.unsqueeze(0)
                x2 = x2.unsqueeze(0)

                x1 = self.spec_augment_op.apply(x1)
                x2 = self.spec_augment_op.apply(x2)

                x1 = x1.squeeze(0)
                x2 = x2.squeeze(0)

        if self.labels is None:
            return x1, x2

        return (x1, x2), self.labels[idx]


class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.99
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


def train_multiple_data(
    title,
    data_source={"covidbreath": 251},
    preprocessing=None,
    strategy="crop",
    dim_fea=1280,
    dim_hidden=1280,
    dim_out=512,
    encoder="efficientnet",
    n_epoches=512,
    training_method="cola",
    augment=True,
    spec_augment=False,
    batch_size=512,
    data_percentage=1.0,
    target_steps=None,
):
    print(data_source)

    method = training_method

    epochs = n_epoches

    print(f"contrastive strategy: {strategy}")

    num_batch = []

    #  constructing dataloaders
    train_loaders, val_loaders = [], []

    print(
        "==============================================================================="
    )
    print("start loading data:")
    for dt, max_len in data_source.items():
        from_npy = True
        if dt in ["covidbreath", "covidcough"]:
            modality = dt[5:]
            filenames = list(
                np.load(
                    "datasets/covid19-sounds/SSL_entireaudio_filenames_{}.npy".format(
                        modality
                    )
                )
            )

        elif dt == "icbhi":
            #  training with audio
            icbhi_filenames = np.load("datasets/icbhi/entire_spec_filenames.npy")
            train_test = np.load("datasets/icbhi/entire_spec_split.npy")
            # exclude testing
            filenames = list(icbhi_filenames[train_test == "train"])

        elif dt == "icbhicycle":
            # training with cycle:
            icbhi_filenames = np.load("datasets/icbhi/cycle_spec_pad2_name.npy")
            train_test = np.load("datasets/icbhi/cycle_spec_split.npy")
            # exclude testing
            filenames = list(icbhi_filenames[train_test == "train"])

        elif dt == "coughvid":
            if preprocessing == "smooth":
                print("using coughvid smooth")

                filenames = list(
                    np.load("datasets/coughvid/entire_spec_filenames_smooth.npy")
                )
            elif preprocessing == "segmented":
                if strategy == "phase":
                    filenames1 = list(
                        np.load(
                            "datasets/coughvid/entire_spec_filenames_segmented_phase_1.npy"
                        )
                    )
                    filenames2 = list(
                        np.load(
                            "datasets/coughvid/entire_spec_filenames_segmented_phase_2.npy"
                        )
                    )
                else:
                    print("using coughvid segmented")
                    filenames = list(
                        np.load("datasets/coughvid/entire_spec_filenames_segmented.npy")
                    )
            else:
                filenames = list(np.load("datasets/coughvid/entire_spec_filenames.npy"))

        elif dt == "hf_lung":
            filenames = list(np.load("datasets/hf_lung/entire_spec_filenames.npy"))

        elif dt == "covidUKexhalation":
            filenames = list(
                np.load("datasets/covidUK/entire_exhalation_filenames.npy")
            )

        elif dt == "covidUKcough":
            if preprocessing == "smooth":
                print("using covidUKcough smooth")
                filenames = list(
                    np.load("datasets/covidUK/entire_cough_filenames_smooth.npy")
                )
            elif preprocessing == "segmented":
                if strategy == "phase":
                    filenames1 = list(
                        np.load(
                            "datasets/covidUK/entire_cough_filenames_segmented_phase_1.npy"
                        )
                    )
                    filenames2 = list(
                        np.load(
                            "datasets/covidUK/entire_cough_filenames_segmented_phase_2.npy"
                        )
                    )
                else:
                    print("using covidUKcough segmented")
                    filenames = list(
                        np.load("datasets/covidUK/entire_cough_filenames_segmented.npy")
                    )
            else:
                filenames = list(np.load("datasets/covidUK/entire_cough_filenames.npy"))

        random.seed(42)

        if strategy == "phase":
            combined_filenames = list(zip(filenames1, filenames2))
            random.shuffle(combined_filenames)
            num_items = int(len(combined_filenames) * data_percentage)
            print("num_items", num_items)

            combined_filenames = combined_filenames[:num_items]
            filenames1, filenames2 = zip(*combined_filenames)

            train1, test1 = train_test_split(
                filenames1, test_size=0.1, random_state=1337
            )
            train2, test2 = train_test_split(
                filenames2, test_size=0.1, random_state=1337
            )

            train_data = AudioDataset(
                train1,
                data_2=train2,
                spec_augment=spec_augment,
                augment=augment,
                from_npy=True,
                max_len=max_len,
                method=method,
                positive_pair_method=strategy,
                preprocessing=preprocessing,
            )
            val_data = AudioDataset(
                test1,
                data_2=test2,
                spec_augment=spec_augment,
                augment=augment,
                from_npy=True,
                max_len=max_len,
                method=method,
                positive_pair_method=strategy,
                preprocessing=preprocessing,
            )

        else:
            random.shuffle(filenames)
            num_items = int(len(filenames) * data_percentage)
            print("num_items", num_items)
            filenames = filenames[:num_items]

            train, test = train_test_split(filenames, test_size=0.1, random_state=1337)

            train_data = AudioDataset(
                train,
                spec_augment=spec_augment,
                augment=augment,
                from_npy=True,
                max_len=max_len,
                method=method,
                positive_pair_method=strategy,
                preprocessing=preprocessing,
                data_percentage=data_percentage,
            )
            val_data = AudioDataset(
                test,
                spec_augment=spec_augment,
                augment=augment,
                from_npy=True,
                max_len=max_len,
                method=method,
                positive_pair_method=strategy,
                preprocessing=preprocessing,
                data_percentage=data_percentage,
            )

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=7
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=True, num_workers=7
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        print(dt, "Length of Training, Validation", len(train_loader), len(val_loader))
        num_batch.append(len(train_loader))

    print(
        "==============================================================================="
    )
    train_loader = combine_dataloaders(train_loaders, train=True)
    val_loader = combine_dataloaders(val_loaders)

    if training_method == "cola":
        model = ColaMD(
            encoder=encoder,
            max_len=data_source,
            dim_fea=dim_fea,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_batch=num_batch,
        )
    logger = CSVLogger(
        save_dir="cks/logs",
        name="combined",
        version=title,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath="cks/model/combined/" + "_".join(data_source.keys()),
        filename="encoder-" + title + "-{epoch:02d}--{valid_acc:.2f}-{valid_loss:.4f}",
        every_n_epochs=20,
        save_top_k=5,
    )

    if target_steps:
        steps_per_epoch = max(num_batch)

        epochs = math.ceil(target_steps / steps_per_epoch)
    print("epochs", epochs)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        precision="16-mixed",
        check_val_every_n_epoch=20,
    )

    print("======================SSL Training==============================")
    trainer.fit(model, train_loader, val_loader)
    print("======================SSL Testing==============================")
    trainer.test(dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str)
    parser.add_argument("--data", type=str, default="multiple")

    # for training with multiple data
    parser.add_argument("--covidbreath", type=bool, default=False)
    parser.add_argument("--covidcough", type=bool, default=False)
    parser.add_argument("--icbhi", type=bool, default=False)
    parser.add_argument("--icbhicycle", type=bool, default=False)
    parser.add_argument("--coughvid", type=bool, default=False)
    parser.add_argument("--hf_lung", type=bool, default=False)
    parser.add_argument("--covidUKexhalation", type=bool, default=False)
    parser.add_argument("--covidUKcough", type=bool, default=False)

    parser.add_argument("--strategy", type=str, default="crop")
    parser.add_argument("--preprocessing", type=str, default=None)
    parser.add_argument("--augment", type=bool, default=False)
    parser.add_argument("--specaugment", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--data_percentage", type=float, default=1.0)
    parser.add_argument("--target_steps", type=float, default=None)

    # control training
    parser.add_argument("--dim_hidden", type=int, default=1280)
    parser.add_argument("--dim_out", type=int, default=512)
    parser.add_argument("--encoder", type=str, default="efficientnet")
    parser.add_argument("--epoches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # training goal
    parser.add_argument("--method", type=str, default="cola")

    args = parser.parse_args()

    optimal_max_len = {
        "covidbreath": 200,
        "covidcough": 50,
        "icbhi": 50,
        "icbhicycle": 50,
        "coughvid": 50,
        "hf_lung": 200,
        "covidUKexhalation": 100,
        "covidUKcough": 50,
    }

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_source = {}
    for dt, max_len in optimal_max_len.items():
        if getattr(args, dt) is True:
            data_source[dt] = max_len
    train_multiple_data(
        args.title,
        data_source=data_source,
        dim_hidden=args.dim_hidden,
        dim_out=args.dim_out,
        encoder=args.encoder,
        n_epoches=args.epoches,
        training_method=args.method,
        strategy=args.strategy,
        preprocessing=args.preprocessing,
        augment=args.augment,
        spec_augment=args.specaugment,
        batch_size=args.batch_size,
        data_percentage=args.data_percentage,
        target_steps=args.target_steps,
    )
