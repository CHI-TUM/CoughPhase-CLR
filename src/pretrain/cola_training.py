# Yuwei (Evelyn) Zhang
# yz798@cam.ac.uk
# Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking
# https://github.com/evelyn0414/OPERA
# some code below is referenced from https://github.com/CVxTz/COLA_pytorch
from autrainer.augmentations import SpecAugment
import os
import argparse
import json
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from lightning.pytorch.utilities import CombinedLoader
import random
from src.util import random_crop, random_mask, random_multiply
from src.model.models_cola import Cola, ColaMD
import math
from bisect import bisect_right


def combine_dataloaders(dataloaders, train=False):
    if train:
        return CombinedLoader(dataloaders, "max_size_cycle")
    return CombinedLoader(dataloaders, "sequential")


class SourceTaggedDataset(ConcatDataset):
    """ConcatDataset whose items also carry the index of the dataset they come from
    and their global index (read by TrainedFileLog)."""

    def __getitem__(self, idx):
        return super().__getitem__(idx), bisect_right(self.cumulative_sizes, idx), idx


def collate_single_source(items):
    return {
        "batch": default_collate([item for item, _, _ in items]),
        "dataset_idx": items[0][1],
        "indices": [idx for _, _, idx in items],  # plain ints, not moved to the GPU
    }


class TrainedFileLog(pl.Callback):
    """
    Records the files of every batch that went through a training step (on_train_batch_end,
    so batches the loader prefetched but never trained on are not counted). Writes to log_dir:
      steps.tsv                  global_step, source, local indices of the batch (one line per step)
      index_<source>.tsv         local index -> file (phase: file1 <tab> file2)
      trained_files_<source>.tsv times trained <tab> file(s), for every file trained on at least once
      summary.json               steps, batches and unique files per source
    The callback only reads the batch, it draws no random numbers.
    """

    def __init__(self, log_dir, dataset, source_names):
        self.log_dir = log_dir
        self.dataset = dataset
        self.source_names = list(source_names)
        self.counts = [np.zeros(len(d), dtype=np.int64) for d in dataset.datasets]
        self.batches = [0] * len(self.source_names)
        self.steps_file = None

    def _files(self, s, i):
        d = self.dataset.datasets[s]
        return [d.data[i]] + ([d.data_2[i]] if d.data_2 is not None else [])

    def on_train_start(self, trainer, pl_module):
        os.makedirs(self.log_dir, exist_ok=True)
        for s, name in enumerate(self.source_names):
            with open(os.path.join(self.log_dir, f"index_{name}.tsv"), "w") as f:
                for i in range(len(self.dataset.datasets[s])):
                    f.write(f"{i}\t" + "\t".join(self._files(s, i)) + "\n")
        self.steps_file = open(os.path.join(self.log_dir, "steps.tsv"), "w")
        self.steps_file.write("global_step\tsource\tlocal_indices\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        s = batch["dataset_idx"]
        offset = self.dataset.cumulative_sizes[s - 1] if s > 0 else 0
        local = [g - offset for g in batch["indices"]]
        assert all(
            0 <= i < len(self.counts[s]) for i in local
        ), "index outside its source"
        np.add.at(self.counts[s], local, 1)
        self.batches[s] += 1
        self.steps_file.write(
            f"{trainer.global_step}\t{self.source_names[s]}\t{','.join(map(str, local))}\n"
        )
        if trainer.global_step % 100 == 0:
            self.steps_file.flush()

    def _write_summary(self, trainer):
        if self.steps_file is None:
            return
        self.steps_file.flush()
        summary = {"global_step": trainer.global_step, "sources": {}}
        for s, name in enumerate(self.source_names):
            with open(
                os.path.join(self.log_dir, f"trained_files_{name}.tsv"), "w"
            ) as f:
                for i in np.flatnonzero(self.counts[s]):
                    f.write(
                        f"{self.counts[s][i]}\t" + "\t".join(self._files(s, i)) + "\n"
                    )
            summary["sources"][name] = {
                "train_split_size": int(len(self.counts[s])),
                "batches": self.batches[s],
                "samples": int(self.counts[s].sum()),
                "unique_files_trained": int((self.counts[s] > 0).sum()),
            }
        with open(os.path.join(self.log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print("trained file log:", self.log_dir, json.dumps(summary["sources"]))

    def on_train_end(self, trainer, pl_module):
        self._write_summary(trainer)
        self.steps_file.close()

    def on_exception(self, trainer, pl_module, exception):
        self._write_summary(trainer)  # keep what was trained before a crash


class SingleSourceBatchSampler(Sampler):
    """
    Each step draws one dataset with probability proportional to its number of
    batches and yields the next batch of that dataset only. This is the sampling
    ColaMD.training_step did on a max_size_cycle CombinedLoader, which also loaded
    (and augmented) a batch of every other dataset per step only to discard it.
    An epoch is max(num_batch) steps, as with max_size_cycle.
    Only full batches are yielded.
    """

    def __init__(self, dataset_sizes, batch_size):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.offsets = np.cumsum([0] + list(dataset_sizes[:-1])).tolist()
        assert min(dataset_sizes) >= batch_size, "every dataset needs a full batch"
        self.num_batch = [n // batch_size for n in dataset_sizes]
        self.streams = [self._batches(i) for i in range(len(dataset_sizes))]

    def _batches(self, i):
        # endless reshuffled passes over dataset i, the short remainder of a pass is dropped
        while True:
            perm = torch.randperm(self.dataset_sizes[i]).tolist()
            for start in range(0, len(perm) - self.batch_size + 1, self.batch_size):
                yield [
                    self.offsets[i] + j for j in perm[start : start + self.batch_size]
                ]

    def __len__(self):
        return max(self.num_batch)

    def __iter__(self):
        sources = range(len(self.dataset_sizes))
        for _ in range(len(self)):
            s = random.choices(sources, weights=self.num_batch, k=1)[0]
            yield next(self.streams[s])


def single_source_loader(datasets, batch_size, num_workers):
    return DataLoader(
        SourceTaggedDataset(datasets),
        batch_sampler=SingleSourceBatchSampler([len(d) for d in datasets], batch_size),
        collate_fn=collate_single_source,
        num_workers=num_workers,
    )


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


# time frames of a phase view (interpolate_to_size), also the crop length of the crop fallback
PHASE_LEN = 50

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
        pair_method = self.positive_pair_method
        if pair_method == "phase" and "_segmented_phase_" not in str(self.data[idx]):
            # a CE recording without detected segments (crop fallback, train_multiple_data):
            # two random crops of its full spectrogram, as for CE
            pair_method = "crop"
            assert (
                self.max_len == PHASE_LEN
            ), "crop fallback views must match the interpolated phase views"
            x = np.load(self.data[idx] + ".npy")

        if pair_method == "phase":
            npy_path1 = self.data[idx]
            npy_path2 = self.data_2[idx]
            x1 = np.load(npy_path1 + ".npy")
            x2 = np.load(npy_path2 + ".npy")
            if os.path.basename(npy_path1) != os.path.basename(npy_path2):
                print("paths not equal: ", npy_path1, npy_path2)

        elif self.positive_pair_method != "phase":
            if self.from_npy:
                npy_path = self.data[idx]
                x = np.load(npy_path + ".npy")
            else:
                x = self.data[idx]

        if self.method == "cola":
            if self.augment:
                x = random_mask(x)

            if pair_method == "phase":
                x1 = interpolate_to_size(x1, target_size=PHASE_LEN)
                x2 = interpolate_to_size(x2, target_size=PHASE_LEN)

            elif pair_method == "crop":
                x1 = random_crop(x, crop_size=self.max_len)
                x2 = random_crop(x, crop_size=self.max_len)

            elif pair_method == "mask":
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


DEFAULT_LR_DECAY_STEPS = 100


def select_recordings(filenames, data_percentage):
    filenames = list(filenames)
    random.seed(42)
    random.shuffle(filenames)
    num_items = int(len(filenames) * data_percentage)
    print("num_items", num_items)
    return train_test_split(filenames[:num_items], test_size=0.1, random_state=1337)


def recording_name(path):
    # segments are named <recording>_cough_<i>
    return os.path.basename(str(path)).rsplit("_cough_", 1)[0]


class LogAmpScale(pl.Callback):
    """Logs the fp16 GradScaler loss scale, which collapses when steps keep overflowing."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            pl_module.log("amp_scale", scaler.get_scale(), on_step=True, on_epoch=False)


class DecayLearningRate(pl.Callback):
    """
    Multiplies the learning rate by 0.99 every `decay_steps` optimizer steps (applied
    smoothly per step), so the schedule depends on the step count only and is the
    same for every data percentage. It used to decay once per epoch, which made
    runs with less data (more, shorter epochs) end at a much smaller learning rate.
    """

    def __init__(self, decay_steps=DEFAULT_LR_DECAY_STEPS):
        self.decay_steps = decay_steps
        self.base_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        self.base_lrs = [
            [param_group["lr"] for param_group in optimizer.param_groups]
            for optimizer in trainer.optimizers
        ]

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        factor = 0.99 ** (trainer.global_step / self.decay_steps)
        for optimizer, base_lrs in zip(trainer.optimizers, self.base_lrs):
            for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
                param_group["lr"] = base_lr * factor


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
    lr_decay_steps=DEFAULT_LR_DECAY_STEPS,
    val_check_steps=1000,
    include_test=(),
    split_dump_dir=None,
    train_file_log=None,
):
    print(data_source)
    if include_test:
        # phase: segmented CLR lists; crop: the unsegmented CE lists (datasets/build_ce_test_pretrain_lists.py)
        assert (strategy == "phase" and preprocessing == "segmented") or (
            strategy == "crop" and preprocessing is None
        ), "include_test: phase+segmented (CLR) or crop+unsegmented (CE) lists only"
        assert set(include_test) <= set(data_source), (include_test, data_source)
        print("INCLUDING benchmark test recordings for:", list(include_test))

    method = training_method

    epochs = n_epoches

    print(f"contrastive strategy: {strategy}")

    num_batch = []

    #  constructing dataloaders
    train_datasets, val_loaders = [], []

    print(
        "==============================================================================="
    )
    print("start loading data:")
    for dt, max_len in data_source.items():
        from_npy = True
        ce_filenames = (
            None  # CE list whose recording selection the CLR phase pairs follow
        )
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
                    # same recordings as the CE list below (datasets/coughvid/get_phase_filenames.py);
                    # with include_test: + the benchmark test recordings CE leaks
                    suffix = "_with_test" if dt in include_test else ""
                    filenames1 = list(
                        np.load(
                            f"datasets/coughvid/entire_spec_filenames_segmented_phase_1{suffix}.npy"
                        )
                    )
                    filenames2 = list(
                        np.load(
                            f"datasets/coughvid/entire_spec_filenames_segmented_phase_2{suffix}.npy"
                        )
                    )
                    ce_filenames = np.load(
                        f"datasets/coughvid/entire_spec_filenames{suffix}.npy"
                    )
                else:
                    print("using coughvid segmented")
                    filenames = list(
                        np.load("datasets/coughvid/entire_spec_filenames_segmented.npy")
                    )
            else:
                # with include_test: clean list + the benchmark test recordings
                # (datasets/build_ce_test_pretrain_lists.py); the CLR phase lists follow this list
                suffix = "_with_test" if dt in include_test else ""
                filenames = list(
                    np.load(f"datasets/coughvid/entire_spec_filenames{suffix}.npy")
                )

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
                    # same recordings as the CE list below (datasets/covidUK/get_phase_filenames.py);
                    # with include_test: + the T2 (covidukcough) test recordings CE leaks
                    suffix = "_with_t2test" if dt in include_test else ""
                    filenames1 = list(
                        np.load(
                            f"datasets/covidUK/entire_cough_filenames_segmented_phase_1{suffix}.npy"
                        )
                    )
                    filenames2 = list(
                        np.load(
                            f"datasets/covidUK/entire_cough_filenames_segmented_phase_2{suffix}.npy"
                        )
                    )
                    ce_filenames = np.load(
                        f"datasets/covidUK/entire_cough_filenames{suffix}.npy"
                    )
                else:
                    print("using covidUKcough segmented")
                    filenames = list(
                        np.load("datasets/covidUK/entire_cough_filenames_segmented.npy")
                    )
            else:
                # with include_test: clean list + the T2 test recordings
                # (datasets/build_ce_test_pretrain_lists.py); the CLR phase lists follow this list
                suffix = "_with_t2test" if dt in include_test else ""
                filenames = list(
                    np.load(f"datasets/covidUK/entire_cough_filenames{suffix}.npy")
                )

        random.seed(42)

        if strategy == "phase" and ce_filenames is not None:
            # the recordings CE trains/validates on at this data_percentage, with all their segment pairs
            # (recordings without segments: crop fallback below)
            ce_train, ce_val = select_recordings(ce_filenames, data_percentage)
            part = {
                **{os.path.basename(str(f)): "train" for f in ce_train},
                **{os.path.basename(str(f)): "val" for f in ce_val},
            }
            pairs = [
                (f1, f2, part.get(recording_name(f1)))
                for f1, f2 in zip(filenames1, filenames2)
            ]
            train1 = [f1 for f1, _, p in pairs if p == "train"]
            train2 = [f2 for _, f2, p in pairs if p == "train"]
            test1 = [f1 for f1, _, p in pairs if p == "val"]
            test2 = [f2 for _, f2, p in pairs if p == "val"]
            print("pairs from CE recordings: train", len(train1), "val", len(test1))
            # crop fallback: a CE recording without detected segments enters once with its full
            # spectrogram (file1 == file2), AudioDataset makes two random crops of it
            segmented = {recording_name(f1) for f1 in filenames1}
            no_segments = {
                part: [
                    str(f) for f in ce_part if os.path.basename(str(f)) not in segmented
                ]
                for part, ce_part in (("train", ce_train), ("val", ce_val))
            }
            train1 += no_segments["train"]
            train2 += no_segments["train"]
            test1 += no_segments["val"]
            test2 += no_segments["val"]
            print(
                "crop fallback (CE recordings without segments): train",
                len(no_segments["train"]),
                "val",
                len(no_segments["val"]),
            )

        elif strategy == "phase":
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

        if strategy == "phase":
            if split_dump_dir:
                os.makedirs(split_dump_dir, exist_ok=True)
                np.savez(
                    os.path.join(split_dump_dir, f"{dt}.npz"),
                    train1=np.array(train1),
                    train2=np.array(train2),
                    val1=np.array(test1),
                    val2=np.array(test2),
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
            train, test = select_recordings(filenames, data_percentage)
            if split_dump_dir:
                os.makedirs(split_dump_dir, exist_ok=True)
                np.savez(
                    os.path.join(split_dump_dir, f"{dt}.npz"),
                    train=np.array(train),
                    val=np.array(test),
                )

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

        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=7,
            drop_last=len(val_data) >= batch_size,
        )

        train_datasets.append(train_data)
        val_loaders.append(val_loader)
        train_batches = len(train_data) // batch_size  # full batches only
        print(dt, "Length of Training, Validation", train_batches, len(val_loader))
        num_batch.append(train_batches)

    print(
        "==============================================================================="
    )
    # same total worker count as before (7 per dataset), now shared by all datasets
    train_loader = single_source_loader(
        train_datasets, batch_size, num_workers=7 * len(train_datasets)
    )
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

    if target_steps:
        schedule = dict(
            max_epochs=-1,
            max_steps=int(target_steps),
            check_val_every_n_epoch=None,
            val_check_interval=int(val_check_steps),
        )
        print(
            "steps",
            int(target_steps),
            "(~%.1f epochs)" % (target_steps / max(num_batch)),
        )
    else:
        schedule = dict(max_epochs=epochs, check_val_every_n_epoch=20)
        print("epochs", epochs)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath="cks/model/combined/" + "_".join(data_source.keys()),
        filename="encoder-"
        + title
        + "-{epoch:02d}-{step}--{valid_acc:.2f}-{valid_loss:.4f}",
        every_n_epochs=(
            None if target_steps else 20
        ),  # with target_steps: every validation
        save_top_k=5,
        save_last=bool(target_steps),
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "encoder-" + title + "-last"

    print(
        "lr decay:",
        (
            f"x0.99 every {lr_decay_steps:g} steps"
            if lr_decay_steps > 0
            else "off (constant lr)"
        ),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[
            # lr_decay_steps <= 0: constant learning rate (no decay)
            *([DecayLearningRate(lr_decay_steps)] if lr_decay_steps > 0 else []),
            LearningRateMonitor(logging_interval="step"),
            LogAmpScale(),
            checkpoint_callback,
            *(
                [
                    TrainedFileLog(
                        train_file_log, train_loader.dataset, data_source.keys()
                    )
                ]
                if train_file_log
                else []
            ),
        ],
        precision="16-mixed",
        **schedule,
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
    parser.add_argument(
        "--lr_decay_steps",
        type=float,
        default=DEFAULT_LR_DECAY_STEPS,
        help="x0.99 every n steps; 0 = constant learning rate",
    )
    parser.add_argument(
        "--include_test",
        nargs="*",
        default=[],
        choices=["coughvid", "covidUKcough"],
        help="leakage experiments: pretrain on lists that contain the benchmark test recordings",
    )
    parser.add_argument(
        "--split_dump_dir",
        type=str,
        default=None,
        help="save each source's train/val filename lists here (for auditing)",
    )
    parser.add_argument(
        "--train_file_log",
        type=str,
        default=None,
        help="record the files of every trained batch here (TrainedFileLog)",
    )
    parser.add_argument(
        "--val_check_steps",
        type=int,
        default=1000,
        help="with --target_steps: validate and checkpoint every n steps",
    )

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
        lr_decay_steps=args.lr_decay_steps,
        val_check_steps=args.val_check_steps,
        include_test=args.include_test,
        split_dump_dir=args.split_dump_dir,
        train_file_log=args.train_file_log,
    )
