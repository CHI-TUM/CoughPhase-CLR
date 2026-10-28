import json
from glob import glob
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.util import train_test_split_from_list, downsample_balanced_dataset
from src.model.models_eval import LinearHead, LinearHeadR
import collections
from tqdm import tqdm


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data[0]
        self.label = data[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        label = self.label[idx]
        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return x, label


class FeatureDatasetR(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data[0]
        self.label = data[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        label = self.label[idx]
        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        return x, label


class DecayLearningRate(pl.Callback):
    def __init__(self, weight=0.97):
        self.old_lrs = []
        self.weight = weight

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
                new_lr = old_lr * self.weight
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


def standardize_splits(x_train, *others):
    """Scale features with mean/std fitted on the training split only (no leakage into val/test)."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    return [scaler.transform(x_train)] + [scaler.transform(x) for x in others]


def ckpt_suffix(standardize):
    # keeps the default checkpoint names unchanged
    return ["std"] if standardize else []


def linear_evaluation_covid19sounds(
    task=1,
    use_feature="opensmile",
    modality="cough",
    l2_strength=1e-4,
    lr=1e-5,
    head="linear",
    batch_size=64,
    epochs=64,
):
    print(
        "linear evaluation ({}) of task".format(head),
        task,
        "with feature set",
        use_feature,
        "modality",
        modality,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "*" * 28,
    )
    folders = {1: "feature/covid19sounds_eval/downsampled/"}
    folder = folders[task]

    x_data = np.load(
        folder + use_feature + "_feature_{}.npy".format(modality)
    ).squeeze()
    y_label = np.load(folder + "labels.npy")
    y_set = np.load(folder + "data_split.npy")

    feat_dim = x_data.shape[1]

    if task == 1:
        x_data_train = x_data[y_set == 0]
        y_label_train = y_label[y_set == 0]
        x_data_vad = x_data[y_set == 1]
        y_label_vad = y_label[y_set == 1]
        x_data_test = x_data[y_set == 2]
        y_label_test = y_label[y_set == 2]
    else:
        raise NotImplementedError(
            f"Task not implemented: Covid-19 sounds task {task}, please check the args."
        )

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_data = FeatureDataset((x_data_train, y_label_train))
    test_data = FeatureDataset((x_data_test, y_label_test))
    val_data = FeatureDataset((x_data_vad, y_label_vad))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength, lr=lr, head=head)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/covidtask" + str(task) + "/" + modality + "/",
        filename="_".join(
            [
                "linear",
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    logger = CSVLogger(
        save_dir="cks/logs",
        name="covid-task" + str(task) + modality,
        version="_".join(
            [
                "linear",
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training linear evaluation ({}) of task".format(head),
        task,
        "with feature set",
        use_feature,
        "modality",
        modality,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "*" * 28,
    )
    return auc


def linear_evaluation_icbhidisease(
    use_feature="opensmile",
    l2_strength=1e-4,
    epochs=64,
    batch_size=64,
    lr=1e-4,
    head="linear",
):
    print("*" * 48)
    print(
        "training dataset icbhi disease using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/icbhidisease_eval/"
    y_set = np.load(feature_dir + "split.npy")
    y_label = np.load(feature_dir + "labels.npy")

    print(collections.Counter(y_label))
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    mask = (y_label == "Healthy") | (y_label == "COPD")
    y_label = y_label[mask]
    y_set = y_set[mask]
    x_data = x_data[mask]

    label_dict = {"Healthy": 0, "COPD": 1}
    y_label = np.array([label_dict[y] for y in y_label])

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)
    feat_dim = x_data.shape[1]

    X_train, X_test, y_train, y_test = train_test_split_from_list(
        x_data, y_label, y_set
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1337, stratify=y_train
    )
    print(collections.Counter(y_train))
    print(collections.Counter(y_val))
    print(collections.Counter(y_test))

    train_data = FeatureDataset((X_train, y_train))
    test_data = FeatureDataset((X_test, y_test))
    val_data = FeatureDataset((X_val, y_val))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )
    loss_func = None

    model = LinearHead(
        feat_dim=feat_dim,
        classes=2,
        l2_strength=l2_strength,
        lr=lr,
        loss_func=loss_func,
        head=head,
    )

    logger = CSVLogger(
        save_dir="cks/logs",
        name="icbhidisease",
        version="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/icbhidisease/",
        filename="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=train_loader)
    trainer.test(dataloaders=val_loader)
    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset icbhi disease using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
    )
    return auc


def linear_evaluation_coughCOPD(
    use_feature="opensmile",
    l2_strength=1e-4,
    epochs=64,
    batch_size=64,
    lr=1e-4,
    head="linear",
):
    print("*" * 48)
    print(
        "training dataset cough_copd disease using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/coughCOPD_eval/"
    y_set = np.load(feature_dir + "split.npy")
    y_label = np.load(feature_dir + "labels.npy")
    print(y_label)
    print(collections.Counter(y_label))
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)
    feat_dim = x_data.shape[1]

    X_train, X_test, y_train, y_test = train_test_split_from_list(
        x_data, y_label, y_set
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1337, stratify=y_train
    )
    print(collections.Counter(y_train))
    print(collections.Counter(y_val))
    print(collections.Counter(y_test))

    train_data = FeatureDataset((X_train, y_train))
    test_data = FeatureDataset((X_test, y_test))
    val_data = FeatureDataset((X_val, y_val))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )
    loss_func = None

    model = LinearHead(
        feat_dim=feat_dim,
        classes=2,
        l2_strength=l2_strength,
        lr=lr,
        loss_func=loss_func,
        head=head,
    )

    logger = CSVLogger(
        save_dir="cks/logs",
        name="icbhidisease",
        version="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/icbhidisease/",
        filename="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=train_loader)
    trainer.test(dataloaders=val_loader)
    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset icbhi disease using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
    )
    return auc


def linear_evaluation_kauh(
    use_feature="opensmile",
    l2_strength=1e-6,
    epochs=64,
    lr=1e-5,
    batch_size=64,
    head="linear",
):
    print("*" * 48)
    print(
        "training dataset kauh using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )
    folder = "feature/kauh_eval/"

    labels = np.load(folder + "labels_both.npy")
    y_set = np.load(folder + "train_test_split.npy")

    x_data = np.load(folder + use_feature + "_feature_both.npy").squeeze()

    label_dict = {"healthy": 0, "asthma": 1, "COPD": 1, "obstructive": 1}
    y_label = np.array([label_dict[y] for y in labels])

    feat_dim = x_data.shape[1]

    X_train, X_test, y_train, y_test = train_test_split_from_list(
        x_data, y_label, y_set
    )
    print("training distribution", collections.Counter(y_train))
    print("testing distribution", collections.Counter(y_test))

    # generate a validation set (seed fixed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=1337, stratify=y_train
    )

    train_data = FeatureDataset((X_train, y_train))
    test_data = FeatureDataset((X_test, y_test))
    val_data = FeatureDataset((X_val, y_val))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=1, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=1, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=1
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength, lr=lr, head=head)

    logger = CSVLogger(
        save_dir="cks/logs",
        name="kauh",
        version="_".join(
            [
                "linear",
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/kauh/",
        filename="_".join(
            [
                "linear",
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
        every_n_epochs=5,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset kauh classes using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
    )
    return auc


def linear_evaluation_coswara(
    use_feature="operaCE1280",
    l2_strength=1e-6,
    epochs=256,
    lr=1e-5,
    batch_size=32,
    modality="cough_shallow",
    label="sex",
    head="linear",
    map_google=False,
    standardize=False,
    return_valid=False,
):
    print("*" * 48)
    print(
        "training dataset coswara of task {} and modality {} using feature extracted by {} with l2_strength {} lr {} head {}".format(
            label, modality, use_feature, l2_strength, lr, head
        )
    )

    feature_dir = "feature/coswara_eval/"

    broad_modality = modality.split("-")[0]
    labels = np.load(
        feature_dir
        + "{}_aligned_{}_label_{}.npy".format(broad_modality, label, modality)
    )
    features = np.load(
        feature_dir + use_feature + "_feature_{}_{}.npy".format(modality, label)
    ).squeeze()
    print(features)
    print(collections.Counter(labels))

    feat_dim = features.shape[1]

    if map_google:
        if "cough" not in modality:
            raise NotImplementedError("not supported")
        split = np.load(feature_dir + "google_{}_{}_split.npy".format(label, modality))
        X_train, X_test, y_train, y_test = train_test_split_from_list(
            features, labels, split
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=1337, stratify=labels
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1337, stratify=y_train
    )

    if use_feature == "vggish":
        # for vggish, this produces a better and more reasonable result, otherwise an extremely low AUROC
        X_train, y_train = downsample_balanced_dataset(X_train, y_train)

    if standardize:
        X_train, X_val, X_test = standardize_splits(X_train, X_val, X_test)

    print(collections.Counter(y_train))
    print(collections.Counter(y_val))
    print(collections.Counter(y_test))

    train_data = FeatureDataset((X_train, y_train))
    test_data = FeatureDataset((X_test, y_test))
    val_data = FeatureDataset((X_val, y_val))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength, lr=lr, head=head)

    logger = CSVLogger(
        save_dir="cks/logs",
        name="coswara",
        version="_".join(
            [
                head,
                label,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/coswara/",
        filename="_".join(
            [
                head,
                label,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
            + ckpt_suffix(standardize)
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset coswara using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "with an auc of",
        auc,
    )
    if return_valid:
        return auc, float(checkpoint_callback.best_model_score)
    return auc


def linear_evaluation_copd_cough(
    use_feature="operaCE1280",
    l2_strength=1e-6,
    epochs=256,
    lr=1e-5,
    batch_size=32,
    head="linear",
    standardize=False,
    return_valid=False,
):
    print("*" * 48)
    print("training dataset custom copd Cough ")

    feature_dir = "feature/custom_copd_eval/"

    labels = np.load(feature_dir + "labels.npy")
    features = np.load(feature_dir + use_feature + "_feature.npy").squeeze()
    print(features)
    print(collections.Counter(labels))

    feat_dim = features.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=1337, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1337, stratify=y_train
    )

    if use_feature == "vggish":
        # for vggish, this produces a better and more reasonable result, otherwise an extremely low AUROC
        X_train, y_train = downsample_balanced_dataset(X_train, y_train)

    if standardize:
        X_train, X_val, X_test = standardize_splits(X_train, X_val, X_test)

    print(collections.Counter(y_train))
    print(collections.Counter(y_val))
    print(collections.Counter(y_test))

    train_data = FeatureDataset((X_train, y_train))
    test_data = FeatureDataset((X_test, y_test))
    val_data = FeatureDataset((X_val, y_val))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength, lr=lr, head=head)

    logger = CSVLogger(
        save_dir="cks/logs",
        name="custom_copd",
        version="_".join(
            [
                head,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/custom_copd/",
        filename="_".join(
            [
                head,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
            + ckpt_suffix(standardize)
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset coswara using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "with an auc of",
        auc,
    )
    if return_valid:
        return auc, float(checkpoint_callback.best_model_score)
    return auc


def linear_evaluation_coughvid(
    use_feature="operaCE1280",
    l2_strength=1e-6,
    epochs=64,
    lr=1e-5,
    batch_size=64,
    label="sex",
    head="linear",
    standardize=False,
    return_valid=False,
):
    print("*" * 48)
    print(
        "training dataset coughvid of task {} and using feature extracted by {} with l2_strength {} lr {}  head".format(
            label, use_feature, l2_strength, lr, head
        )
    )

    feature_dir = "feature/coughvid_eval/"

    y_set = np.load(feature_dir + "split_{}.npy".format(label))
    y_label = np.load(feature_dir + "label_{}.npy".format(label))
    print(collections.Counter(y_label))
    x_data = np.load(
        feature_dir + use_feature + "_feature_{}.npy".format(label)
    ).squeeze()

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)

    feat_dim = x_data.shape[1]

    x_data_train = x_data[y_set == "train"]
    y_label_train = y_label[y_set == "train"]
    x_data_vad = x_data[y_set == "val"]
    y_label_vad = y_label[y_set == "val"]
    x_data_test = x_data[y_set == "test"]
    y_label_test = y_label[y_set == "test"]

    if standardize:
        x_data_train, x_data_vad, x_data_test = standardize_splits(
            x_data_train, x_data_vad, x_data_test
        )

    train_data = FeatureDataset((x_data_train, y_label_train))
    test_data = FeatureDataset((x_data_test, y_label_test))
    val_data = FeatureDataset((x_data_vad, y_label_vad))

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength, lr=lr)

    logger = CSVLogger(
        save_dir="cks/logs",
        name="coughvid",
        version="_".join(
            [
                head,
                label,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/coughvid/",
        filename="_".join(
            [
                head,
                label,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
            + ckpt_suffix(standardize)
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset coughvid of task {} and using feature extracted by {} with l2_strength {} lr {} ".format(
            label, use_feature, l2_strength, lr
        )
    )
    if return_valid:
        return auc, float(checkpoint_callback.best_model_score)
    return auc


def linear_evaluation_coviduk(
    use_feature="operaCE1280",
    l2_strength=1e-6,
    epochs=64,
    lr=1e-5,
    batch_size=64,
    modality="exhalation",
    head="linear",
    standardize=False,
    return_valid=False,
):
    print("*" * 48)
    print(
        "training dataset covidUK of {} and using feature extracted by {} with l2_strength {} lr {}  head".format(
            modality, use_feature, l2_strength, lr, head
        )
    )

    feature_dir = "feature/coviduk_eval/"

    y_set = np.load(feature_dir + "split_{}.npy".format(modality))
    y_label = np.load(feature_dir + "label_{}.npy".format(modality))
    print(collections.Counter(y_label))
    x_data = np.load(
        feature_dir + use_feature + "_feature_{}.npy".format(modality)
    ).squeeze()

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)

    feat_dim = x_data.shape[1]

    x_data_train = x_data[y_set == "train"]
    y_label_train = y_label[y_set == "train"]
    x_data_vad = x_data[y_set == "val"]
    y_label_vad = y_label[y_set == "val"]
    x_data_test = x_data[y_set == "test"]
    y_label_test = y_label[y_set == "test"]

    if standardize:
        x_data_train, x_data_vad, x_data_test = standardize_splits(
            x_data_train, x_data_vad, x_data_test
        )

    train_data = FeatureDataset((x_data_train, y_label_train))
    test_data = FeatureDataset((x_data_test, y_label_test))
    val_data = FeatureDataset((x_data_vad, y_label_vad))

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(feat_dim=feat_dim, classes=2, l2_strength=l2_strength, lr=lr)

    logger = CSVLogger(
        save_dir="cks/logs",
        name="coviduk",
        version="_".join(
            [
                head,
                modality,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/coviduk/",
        filename="_".join(
            [
                head,
                modality,
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
            + ckpt_suffix(standardize)
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset covidUK of {} and using feature extracted by {} with l2_strength {} lr {}  head".format(
            modality, use_feature, l2_strength, lr, head
        )
    )
    if return_valid:
        return auc, float(checkpoint_callback.best_model_score)
    return auc


# padding length (input_sec of get_entire_signal_librosa) each eval set's features are extracted with;
# features with another padding length are stored as <feature>_sec<N>_feature_...
NATIVE_SEC = {
    "coviduk": 2,
    "coughvidcovid": 2,
    "coughvidsex": 2,
    "coswaracovid": 8,
    "coswarasex": 8,
}


def eval_feature_name(use_feature, dataset, input_sec=None):
    if input_sec is None or input_sec == NATIVE_SEC[dataset]:
        return use_feature
    return "{}_sec{}".format(use_feature, input_sec)


def load_eval_splits(dataset, use_feature, input_sec=None):
    """Train / val / test (features, labels, file names) of a binary eval set, split exactly as in its
    linear_evaluation_* function. input_sec selects features extracted with another padding length."""
    feature = eval_feature_name(use_feature, dataset, input_sec)

    if dataset in ["coviduk", "coughvidcovid", "coughvidsex"]:
        feature_dir, key = {
            "coviduk": ("feature/coviduk_eval/", "cough"),
            "coughvidcovid": ("feature/coughvid_eval/", "covid"),
            "coughvidsex": ("feature/coughvid_eval/", "gender"),
        }[dataset]
        y_set = np.load(feature_dir + "split_{}.npy".format(key))
        y_label = np.load(feature_dir + "label_{}.npy".format(key))
        files = np.load(feature_dir + "sound_dir_loc_{}.npy".format(key))
        x_data = np.load(feature_dir + feature + "_feature_{}.npy".format(key)).squeeze()
        if use_feature == "vggish":
            x_data = np.nan_to_num(x_data)
        assert len(x_data) == len(y_label) == len(files)
        return {
            s: (x_data[y_set == s], y_label[y_set == s], files[y_set == s])
            for s in ["train", "val", "test"]
        }

    # coswara cough-shallow; COVID-19 reuses the files and features of the sex list
    feature_dir, modality = "feature/coswara_eval/", "cough-shallow"
    files = np.load(feature_dir + "cough_aligned_filenames_sex_w_{}.npy".format(modality))
    features = np.load(
        feature_dir + feature + "_feature_{}_sex.npy".format(modality)
    ).squeeze()
    if dataset == "coswarasex":
        labels = np.load(feature_dir + "cough_aligned_sex_label_{}.npy".format(modality))
        # same runtime split as linear_evaluation_coswara (splitting indices permutes like the arrays)
        idx = np.arange(len(labels))
        idx_train, idx_test = train_test_split(
            idx, test_size=0.2, random_state=1337, stratify=labels
        )
        idx_train, idx_val = train_test_split(
            idx_train, test_size=0.2, random_state=1337, stratify=labels[idx_train]
        )
        split = {"train": idx_train, "val": idx_val, "test": idx_test}
    else:
        labels = np.load(feature_dir + "cough_aligned_covid_label_{}.npy".format(modality))
        y_set = np.load(feature_dir + "cough_aligned_covid_split_{}.npy".format(modality))
        split = {s: np.where(y_set == s)[0] for s in ["train", "val", "test"]}
    assert len(features) == len(labels) == len(files)
    return {s: (features[i], labels[i], files[i]) for s, i in split.items()}


def linear_evaluation_cross(
    train_on,
    test_on,
    use_feature="operaCE1280",
    l2_strength=1e-5,
    epochs=64,
    lr=1e-4,
    batch_size=64,
    head="linear",
    standardize=False,
    return_valid=False,
    match_preprocessing=True,
):
    """Cross-dataset evaluation: train the head on the train split of train_on (checkpoint picked on its
    val split) and test it on the test split of test_on. The training side always uses its own features;
    with match_preprocessing the test features are extracted with train_on's padding length."""
    test_sec = NATIVE_SEC[train_on] if match_preprocessing else None
    print("*" * 48)
    print(
        "cross-dataset: train {} (feature {}) -> test {} (feature {}) with l2_strength {} lr {} head {}".format(
            train_on,
            eval_feature_name(use_feature, train_on),
            test_on,
            eval_feature_name(use_feature, test_on, test_sec),
            l2_strength,
            lr,
            head,
        )
    )

    source = load_eval_splits(train_on, use_feature)
    target = load_eval_splits(test_on, use_feature, input_sec=test_sec)
    X_train, y_train, _ = source["train"]
    X_val, y_val, _ = source["val"]
    X_test, y_test, _ = target["test"]
    for y in [y_train, y_val, y_test]:
        assert set(np.unique(y)) == {0, 1}, collections.Counter(y)

    if standardize:
        # fitted on the training split of train_on only
        X_train, X_val, X_test = standardize_splits(X_train, X_val, X_test)

    print(collections.Counter(y_train))
    print(collections.Counter(y_val))
    print(collections.Counter(y_test))

    train_loader = DataLoader(
        FeatureDataset((X_train, y_train)), batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        FeatureDataset((X_val, y_val)), batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        FeatureDataset((X_test, y_test)), batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(
        feat_dim=X_train.shape[1], classes=2, l2_strength=l2_strength, lr=lr, head=head
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/cross/",
        filename="_".join(
            [
                head,
                train_on,
                "to",
                test_on,
                "matched" if match_preprocessing else "unmatched",
                use_feature,
                str(batch_size),
                str(lr),
                str(epochs),
                str(l2_strength),
            ]
            + ckpt_suffix(standardize)
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished cross-dataset train {} -> test {} using feature extracted by {} with an auc of".format(
            train_on, test_on, use_feature
        ),
        auc,
    )
    if return_valid:
        return auc, float(checkpoint_callback.best_model_score)
    return auc


def linear_evaluation_ssbpr(
    n_cls=5,
    use_feature="opensmile",
    l2_strength=1e-5,
    epochs=64,
    batch_size=64,
    lr=1e-4,
    head="linear",
    seed=None,
    five_fold=False,
    split_fold=False,
):
    print("*" * 48)
    print(
        "training dataset SSBPR using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/snoring_eval/"

    y_label = np.load(feature_dir + "labels.npy")
    print(collections.Counter(y_label))
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)

    feat_dim = x_data.shape[1]
    print(feat_dim)

    train_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2

    seed = 42
    _x_train, x_data_test, _y_train, y_label_test = train_test_split(
        x_data, y_label, test_size=test_ratio, random_state=seed, stratify=y_label
    )

    x_data_train, x_data_vad, y_label_train, y_label_vad = train_test_split(
        _x_train,
        _y_train,
        test_size=validation_ratio / (validation_ratio + train_ratio),
        random_state=seed,
        stratify=_y_train,
    )

    print(collections.Counter(y_label_train))
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))

    train_data = FeatureDataset((x_data_train, y_label_train))
    test_data = FeatureDataset((x_data_test, y_label_test))
    val_data = FeatureDataset((x_data_vad, y_label_vad))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LinearHead(
        feat_dim=feat_dim, classes=n_cls, l2_strength=l2_strength, lr=lr, head=head
    )

    logger = CSVLogger(
        save_dir="cks/logs",
        name="ssbpr",
        version="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        ),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_auc",
        mode="max",
        dirpath="cks/linear/ssbpr/",
        filename="_".join(
            [head, use_feature, str(batch_size), str(lr), str(epochs), str(l2_strength)]
        )
        + "-{epoch:02d}-{valid_auc:.2f}",
        every_n_epochs=3,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        # logger=logger,
        logger=False,
        callbacks=[DecayLearningRate(), checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=train_loader)
    trainer.test(dataloaders=val_loader)
    test_res = trainer.test(dataloaders=test_loader)
    auc = test_res[0]["test_auc"]
    print(
        "finished training dataset SSBPR using feature extracted by " + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )
    return auc


def linear_evaluation_mmlung(
    use_feature="opensmile",
    method="LOOCV",
    l2_strength=1e-5,
    epochs=64,
    lr=1e-5,
    batch_size=40,
    modality="cough",
    label="FVC",
    head="mlp",
):
    from sklearn.preprocessing import StandardScaler

    print("*" * 48)
    print(
        "training dataset MMLung using feature extracted by " + use_feature,
        "By sklearn",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/mmlung_eval/"

    y_label = np.load(feature_dir + "label.npy")
    if label == "FVC":
        y_label = y_label[:, 0]
    if label == "FEV1":
        y_label = y_label[:, 1]
    if label == "FEV1_FVC":
        y_label = y_label[:, 2]

    if modality == "breath":
        x_data = np.load(
            feature_dir + "Deep_Breath_file_" + use_feature + "_feature.npy"
        ).squeeze()

    if modality == "vowels":
        x_data = np.load(
            feature_dir + "O_Single_file_" + use_feature + "_feature.npy"
        ).squeeze()

    if use_feature == "vggish":
        x_data = np.nan_to_num(x_data)

    print(label, "distribution:", np.mean(y_label), np.std(y_label))
    y_label = y_label.reshape((-1, 1))

    # # leave one out cross validation
    MAEs = []
    MAPEs = []
    for s in tqdm(range(40)):
        x_test, y_test = x_data[s], y_label[s]
        x_test = x_test.reshape(1, -1)
        y_test = y_test.reshape(1, -1)

        X_train = np.delete(x_data, s, axis=0)
        y_train = np.delete(y_label, s, axis=0)

        if "opensmile" in use_feature:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            x_test = scaler.transform(x_test)

        x_train, x_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.11, random_state=42
        )

        train_mean, train_std = np.mean(y_train), np.std(y_train)
        print("mean and std of training data:", train_mean, train_std)

        train_data = FeatureDatasetR((x_train, y_train))
        test_data = FeatureDatasetR((x_test, y_test))
        val_data = FeatureDatasetR((x_val, y_val))

        train_loader = DataLoader(
            train_data, batch_size=batch_size, num_workers=8, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, num_workers=4, shuffle=False
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=4
        )

        feat_dim = x_data.shape[1]
        model = LinearHeadR(
            feat_dim=feat_dim,
            output_dim=1,
            l2_strength=l2_strength,
            head=head,
            mean=train_mean,
            std=train_std,
        )

        logger = CSVLogger(
            save_dir="cks/logs",
            name="mmlung",
            version="_".join(
                [
                    head,
                    use_feature,
                    str(batch_size),
                    str(lr),
                    str(epochs),
                    str(l2_strength),
                ]
            ),
        )

        early_stop_callback = EarlyStopping(
            monitor="valid_MAE", min_delta=0.001, patience=5, verbose=True, mode="min"
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="valid_MAE",
            mode="min",
            dirpath="cks/linear/mmlung/",
            filename="_".join(
                [
                    head,
                    use_feature,
                    str(batch_size),
                    str(lr),
                    str(epochs),
                    str(l2_strength),
                ]
            )
            + "-{epoch:02d}-{valid_MAE:.3f}",
            every_n_epochs=3,
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            devices=1,
            # logger=logger,
            logger=False,
            callbacks=[
                DecayLearningRate(weight=0.97),
                checkpoint_callback,
                early_stop_callback,
            ],
            gradient_clip_val=1.0,
            log_every_n_steps=1,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_loader, val_loader)

        test_res = trainer.test(dataloaders=test_loader)
        mae, mape = test_res[0]["test_MAE"], test_res[0]["test_MAPE"]

        MAEs.append(mae)
        MAPEs.append(mape)

    return MAEs, MAPEs


def linear_evaluation_nosemic(
    use_feature="opensmile",
    method="LOOCV",
    l2_strength=1e-5,
    epochs=64,
    batch_size=32,
    lr=1e-4,
    head="mlp",
):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("*" * 48)
    print(
        "training dataset Nose Breathing audio using feature extracted by "
        + use_feature,
        "with l2_strength",
        l2_strength,
        "lr",
        lr,
        "head",
        head,
    )

    feature_dir = "feature/nosemic_eval/"

    uids = np.load(feature_dir + "uids.npy")
    y_label = np.load(feature_dir + "labels.npy")
    y_label = np.array([float(y) for y in y_label]).reshape(-1, 1)
    print("labels:", y_label)
    x_data = np.load(feature_dir + use_feature + "_feature.npy").squeeze()

    feat_dim = x_data.shape[1]
    print(feat_dim)
    print(uids.shape, x_data.shape)

    MAEs = []
    MAPEs = []
    for uid in [
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "17",
        "18",
        "19",
        "20",
        "21",
    ]:

        x_train = x_data[uids != uid, :]
        x_test = x_data[uids == uid, :]
        y_train = y_label[uids != uid]
        y_test = y_label[uids == uid]

        if "opensmile" in use_feature:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        train_mean, train_std = np.mean(y_train), np.std(y_train)
        print("mean and std of training data:", train_mean, train_std)

        train_data = FeatureDatasetR((x_train, y_train))
        test_data = FeatureDatasetR((x_test, y_test))
        val_data = FeatureDatasetR((x_val, y_val))

        train_loader = DataLoader(
            train_data, batch_size=batch_size, num_workers=8, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, num_workers=4, shuffle=False
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=4
        )

        model = LinearHeadR(
            feat_dim=feat_dim,
            output_dim=1,
            l2_strength=l2_strength,
            head=head,
            mean=train_mean,
            std=train_std,
        )

        logger = CSVLogger(
            save_dir="cks/logs",
            name="nosemic",
            version="_".join(
                [
                    head,
                    use_feature,
                    str(batch_size),
                    str(lr),
                    str(epochs),
                    str(l2_strength),
                ]
            ),
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="valid_MAE",
            mode="min",
            dirpath="cks/linear/nosemic/",
            filename="_".join(
                [
                    head,
                    use_feature,
                    str(batch_size),
                    str(lr),
                    str(epochs),
                    str(l2_strength),
                ]
            )
            + "-{epoch:02d}-{valid_MAE:.3f}",
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            devices=1,
            # logger=logger,
            logger=False,
            callbacks=[DecayLearningRate(weight=0.97), checkpoint_callback],
            gradient_clip_val=1.0,
            log_every_n_steps=1,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_loader, val_loader)

        test_res = trainer.test(dataloaders=test_loader)
        mae, mape = test_res[0]["test_MAE"], test_res[0]["test_MAPE"]

        MAEs.append(mae)
        MAPEs.append(mape)

    return MAEs, MAPEs


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()

    # these args need to be entered according to tasks
    parser.add_argument("--task", type=str, default="coughvidsex")
    parser.add_argument("--label", type=str, default="sex")  # prediction target
    parser.add_argument("--modality", type=str, default="cough-shallow")
    parser.add_argument("--pretrain", type=str, default="opera_task_20")
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--LOOCV", type=bool, default=False)

    # these can follow default
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--l2_strength", type=float, default=1e-5)
    parser.add_argument("--head", type=str, default="linear")
    parser.add_argument(
        "--mapgoogle", type=bool, default=False
    )  # align test set with HeAR
    parser.add_argument("--n_run", type=int, default=5)

    args = parser.parse_args()

    feature = args.pretrain
    # if feature not in ["vggish", "opensmile", "clap", "audiomae"]:  # baselines
    #     feature += str(args.dim)

    if not args.LOOCV:
        # report mean and std for 5 runs with random seeds
        auc_scores = []
        for seed in range(args.n_run):

            # fix seeds for reproducibility
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # if args.task == "coughCOPD":
            #     auc = linear_evaluation_coughCOPD(
            #         use_feature=feature,
            #         epochs=64,
            #         batch_size=32,
            #         l2_strength=args.l2_strength,
            #         lr=args.lr,
            #         head=args.head,
            #     )
            if args.task == "customCoughCOPD":
                auc = linear_evaluation_copd_cough(
                    use_feature=feature,
                    epochs=64,
                    batch_size=32,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                )
            elif args.task == "covid19sounds":
                auc = linear_evaluation_covid19sounds(
                    1,
                    feature,
                    modality=args.modality,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                )
            elif args.task == "icbhidisease":
                auc = linear_evaluation_icbhidisease(
                    use_feature=feature,
                    epochs=64,
                    batch_size=32,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                )
            elif args.task == "kauh":
                auc = linear_evaluation_kauh(
                    use_feature=feature,
                    epochs=50,
                    batch_size=32,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                )
            elif args.task == "coswarasmoker":
                auc = linear_evaluation_coswara(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    batch_size=32,
                    lr=args.lr,
                    modality=args.modality,
                    label="smoker",
                    head=args.head,
                )
            elif args.task == "coswarasex":
                auc = linear_evaluation_coswara(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    batch_size=32,
                    lr=args.lr,
                    modality=args.modality,
                    label="sex",
                    head=args.head,
                )
            # elif args.task == "copd":
            #     auc = linear_evaluation_copd(
            #         use_feature=feature,
            #         l2_strength=args.l2_strength,
            #         lr=args.lr,
            #         head=args.head,
            #         epochs=64,
            #     )
            elif args.task == "coughvidcovid":
                auc = linear_evaluation_coughvid(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    batch_size=64,
                    label="covid",
                    head=args.head,
                )
            elif args.task == "coughvidsex":
                auc = linear_evaluation_coughvid(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    batch_size=64,
                    label="gender",
                    head=args.head,
                )
            elif args.task == "coviduk":
                auc = linear_evaluation_coviduk(
                    use_feature=feature,
                    epochs=64,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    batch_size=64,
                    modality=args.modality,
                    head=args.head,
                )
            elif args.task == "snoring":
                auc = linear_evaluation_ssbpr(
                    use_feature=feature,
                    l2_strength=args.l2_strength,
                    lr=args.lr,
                    head=args.head,
                    epochs=32,
                    seed=seed,
                )
            auc_scores.append(auc)
        print("=" * 48)
        print(auc_scores)
        print(
            "Five times mean task {} feature {} results: auc mean {:.3f} ± {:.3f}".format(
                args.task, feature, np.mean(auc_scores), np.std(auc_scores)
            )
        )
        print("=" * 48)
    else:
        # Leave one out cross validation

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        if args.task == "spirometry":
            maes, mapes = linear_evaluation_mmlung(
                use_feature=feature,
                method="LOOCV",
                l2_strength=1e-1,
                epochs=64,
                lr=1e-1,
                batch_size=64,
                modality=args.modality,
                label=args.label,
                head=args.head,
            )

        if args.task == "rr":
            maes, mapes = linear_evaluation_nosemic(
                use_feature=feature,
                method="LOOCV",
                l2_strength=1e-1,
                epochs=64,
                batch_size=64,
                lr=1e-4,
                head=args.head,
            )

        print("=" * 48)
        print(maes)
        print(mapes)
        print(
            "Five times mean task {} feature {} results: MAE mean {:.3f} ± {:.3f}".format(
                args.task, feature, np.mean(maes), np.std(maes)
            )
        )
        print(
            "Five times mean task {} feature {} results: MAPE mean {:.3f} ± {:.3f}".format(
                args.task, feature, np.mean(mapes), np.std(mapes)
            )
        )
        print("=" * 48)
