"""Optional professional neural engine backed by real PyTorch training.

The educational NumPy engine in :mod:`neural_forge.nn` remains unchanged. This module
is imported safely when PyTorch is absent; callers receive an explicit availability
error instead of a fake result.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import datasets
from . import nn as educational_nn

try:  # PyTorch is an optional, sizeable dependency.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - availability path is tested without importing torch internals
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class TorchUnavailableError(RuntimeError):
    pass


def availability() -> dict[str, Any]:
    if torch is None:
        return {"installed": False, "version": None, "cuda_available": False, "cuda_devices": 0}
    return {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_devices": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def _require_torch() -> None:
    if torch is None:
        raise TorchUnavailableError(
            "PyTorch is not installed. Install backend/requirements-torch.txt, then restart NEURAL FORGE."
        )


def _device(requested: str) -> tuple[Any, list[str]]:
    _require_torch()
    warnings: list[str] = []
    requested = requested if requested in {"auto", "cpu", "cuda"} else "auto"
    if requested == "cuda" and not torch.cuda.is_available():
        warnings.append("CUDA was requested but is unavailable; training fell back to CPU.")
        return torch.device("cpu"), warnings
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), warnings
    return torch.device(requested), warnings


def _seed_everything(seed: int) -> None:
    _require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is preferable to maximum throughput in a teaching lab.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass


def _activation(name: str) -> Any:
    choices = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
    }
    if name not in choices:
        raise ValueError(f"Unknown activation '{name}'.")
    return choices[name]


if nn is not None:
    class ConfigurableMLP(nn.Module):
        """Small configurable classifier used by the professional tabular lab."""

        def __init__(self, input_size: int, hidden: list[int], output_size: int, activation: str, dropout: float):
            super().__init__()
            act = _activation(activation)
            layers: list[nn.Module] = []
            sizes = [input_size, *hidden, output_size]
            for index, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
                layers.append(nn.Linear(fan_in, fan_out))
                if index < len(sizes) - 2:
                    layers.append(act())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
            self.network = nn.Sequential(*layers)

        def forward(self, inputs: Any) -> Any:
            return self.network(inputs)


    class DigitCNN(nn.Module):
        """CNN sized for scikit-learn's local 8×8 digit images."""

        def __init__(self, dropout: float = 0.1):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(16 * 2 * 2, 10))

        def forward(self, inputs: Any) -> Any:
            return self.classifier(self.features(inputs))
else:  # Keep names importable for documentation/type consumers when torch is absent.
    ConfigurableMLP = None
    DigitCNN = None


def _optimizer(name: str, parameters: Any, learning_rate: float, weight_decay: float) -> Any:
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'.")


def _epoch(
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: Any,
    optimizer: Any | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            count += batch
    return total_loss / max(count, 1), correct / max(count, 1)


def _history_template() -> dict[str, list[Any]]:
    return {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


def _append_history(history: dict[str, list[Any]], epoch: int, train: tuple[float, float], val: tuple[float, float]) -> None:
    history["epoch"].append(epoch)
    history["train_loss"].append(round(train[0], 6))
    history["train_acc"].append(round(train[1], 6))
    history["val_loss"].append(round(val[0], 6))
    history["val_acc"].append(round(val[1], 6))


def _save_checkpoint(model: Any, path: Path | None, metadata: dict[str, Any]) -> dict[str, Any] | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only checkpoints produced by this process are written. There is deliberately no
    # endpoint for loading arbitrary uploaded .pt/.pkl files.
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temp)
    os.replace(temp, path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"id": path.stem, "format": "pytorch-state-dict", "metadata_file": meta_path.name}


def train_tabular(config: dict[str, Any], checkpoint_path: Path | None = None, frame: Any | None = None) -> dict[str, Any]:
    """Train a genuine configurable classification MLP.

    Bundled numeric datasets preserve the NumPy-lab comparison path. A validated
    player-owned frame may also be supplied; its train-fitted preprocessing handles
    numeric and bounded categorical features without serializing an unsafe pipeline.
    """
    _require_torch()
    started = time.perf_counter()
    dataset_name = str(config.get("dataset", "moons"))
    personal = frame is not None
    preprocessing_summary: dict[str, Any]
    if personal:
        target = str(config.get("target", ""))
        if not target or target not in frame.columns:
            raise ValueError("Choose a target column from the uploaded dataset.")
        requested = config.get("features")
        features = [str(value) for value in requested] if isinstance(requested, list) else [str(column) for column in frame.columns if column != target]
        features = list(dict.fromkeys(features))
        if not features or len(features) > 100 or target in features or any(column not in frame.columns for column in features):
            raise ValueError("Choose 1–100 valid feature columns that do not include the target.")
        usable = frame.loc[frame[target].notna(), [*features, target]].copy()
        if len(usable) < 20:
            raise ValueError("At least 20 rows with a non-missing target are required for PyTorch training.")
        labels = usable[target].astype(str)
        classes = sorted(labels.unique().tolist())
        if not 2 <= len(classes) <= 100:
            raise ValueError("PyTorch classification requires 2–100 target classes.")
        class_to_index = {label: index for index, label in enumerate(classes)}
        y = labels.map(class_to_index).to_numpy(dtype="int64")
        counts = np.bincount(y)
        if counts.min() < 2:
            raise ValueError("Every target class needs at least two rows for a stratified validation split.")
        numeric_features = [column for column in features if usable[column].dtype.kind in "biufc"]
        categorical_features = [column for column in features if column not in numeric_features]
        estimated_width = len(numeric_features) + sum(min(int(usable[column].nunique(dropna=True)), 200) for column in categorical_features)
        if estimated_width > 2_048:
            raise ValueError("Categorical features would expand beyond 2,048 inputs. Remove high-cardinality columns.")
        raw_features = usable[features]
    else:
        if dataset_name not in educational_nn.NN_DATASETS:
            raise ValueError("Unknown bundled dataset. Use a player-owned dataset identifier with target/features for custom training.")
        X, y, features, classes = educational_nn.load_nn_data(dataset_name)
        numeric_features, categorical_features = list(features), []
        raw_features = None
    if len(classes) < 2:
        raise ValueError("The target must contain at least two classes.")

    hidden = [int(np.clip(v, 1, 256)) for v in config.get("hidden", [32, 16])][:4]
    if not hidden:
        hidden = [16]
    activation = str(config.get("activation", "relu"))
    _activation(activation)  # validate before constructing the model
    optimizer_name = str(config.get("optimizer", "adam"))
    learning_rate = float(np.clip(config.get("learning_rate", config.get("lr", 0.001)), 1e-6, 1.0))
    batch_size = int(np.clip(config.get("batch_size", 32), 1, 512))
    epochs = int(np.clip(config.get("epochs", 30), 1, 200))
    dropout = float(np.clip(config.get("dropout", 0.1), 0.0, 0.8))
    weight_decay = float(np.clip(config.get("weight_decay", 0.0), 0.0, 1.0))
    seed = int(np.clip(config.get("seed", 42), 0, 2**31 - 1))
    validation_size = float(np.clip(config.get("validation_size", 0.2), 0.1, 0.5))
    device, warnings = _device(str(config.get("device", "auto")))
    _seed_everything(seed)

    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=validation_size, random_state=seed, stratify=y
    )
    if personal:
        transformers = []
        if numeric_features:
            transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features))
        if categorical_features:
            transformers.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=200))]), categorical_features))
        preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
        X_train = np.asarray(preprocessor.fit_transform(raw_features.iloc[train_idx]), dtype="float32")
        X_val = np.asarray(preprocessor.transform(raw_features.iloc[val_idx]), dtype="float32")
        if X_train.shape[1] < 1 or X_train.shape[1] > 2_048:
            raise ValueError("Transformed feature width must be between 1 and 2,048.")
        output_features = [str(value) for value in preprocessor.get_feature_names_out()]
        preprocessing_summary = {
            "fit_scope": "training_rows_only", "numeric": numeric_features, "categorical": categorical_features,
            "output_features": output_features, "output_width": len(output_features),
            "note": "Checkpoint contains model weights; preprocessing metadata is descriptive and no unsafe pickle is stored.",
        }
    else:
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx]).astype("float32")
        X_val = scaler.transform(X[val_idx]).astype("float32")
        output_features = list(features)
        preprocessing_summary = {
            "fit_scope": "training_rows_only", "numeric": list(features), "categorical": [],
            "output_features": output_features, "output_width": len(output_features),
        }
    y_train, y_val = y[train_idx].astype("int64"), y[val_idx].astype("int64")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ConfigurableMLP(X_train.shape[1], hidden, len(classes), activation, dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = _optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)
    history = _history_template()
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, loss_fn, device, optimizer)
        # _epoch calls model.train(False) and torch.no_grad() for validation.
        val_metrics = _epoch(model, val_loader, loss_fn, device, None)
        _append_history(history, epoch, train_metrics, val_metrics)

    best_epoch = int(np.argmin(history["val_loss"])) + 1
    duration = time.perf_counter() - started
    final = {
        "train_loss": history["train_loss"][-1],
        "validation_loss": history["val_loss"][-1],
        "train_accuracy": history["train_acc"][-1],
        "validation_accuracy": history["val_acc"][-1],
    }
    resolved = {
        "dataset": dataset_name,
        "source": "player_upload" if personal else "bundled",
        "target": str(config.get("target")) if personal else None,
        "features": list(features),
        "preprocessing": preprocessing_summary,
        "hidden": hidden,
        "activation": activation,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "seed": seed,
        "validation_size": validation_size,
        "device_requested": str(config.get("device", "auto")),
    }
    checkpoint = _save_checkpoint(
        model,
        checkpoint_path,
        {"engine": "pytorch", "task": "classification", "classes": classes, "features": features, "preprocessing": preprocessing_summary, "config": resolved},
    )
    return {
        "engine": "pytorch",
        "task": "classification",
        "device": str(device),
        "torch_version": str(torch.__version__),
        "duration_seconds": round(duration, 4),
        "seed": seed,
        "config": resolved,
        "features": list(features),
        "transformed_features": output_features,
        "preprocessing": preprocessing_summary,
        "classes": classes,
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "final": final,
        "best_validation_epoch": best_epoch,
        "warnings": warnings,
        "checkpoint": checkpoint,
    }


def _augment_digits(images: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """Apply a bounded deterministic augmentation to training images only."""
    if mode not in {"none", "shift", "noise", "shift_noise"}:
        raise TorchTrainingError("augmentation must be none, shift, noise, or shift_noise")
    output = images.copy()
    if mode == "none":
        return output
    rng = np.random.default_rng(seed)
    if "shift" in mode:
        shifted = np.zeros_like(output)
        for index, image in enumerate(output):
            dy, dx = (int(value) for value in rng.integers(-1, 2, size=2))
            source_y = slice(max(0, -dy), min(8, 8 - dy))
            source_x = slice(max(0, -dx), min(8, 8 - dx))
            target_y = slice(max(0, dy), min(8, 8 + dy))
            target_x = slice(max(0, dx), min(8, 8 + dx))
            shifted[index, :, target_y, target_x] = image[:, source_y, source_x]
        output = shifted
    if "noise" in mode:
        output = np.clip(output + rng.normal(0, 0.05, size=output.shape).astype("float32"), 0, 1)
    return output.astype("float32")


def train_cnn(config: dict[str, Any], checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Train a real Conv2D network on the local 8×8 digits dataset."""
    _require_torch()
    started = time.perf_counter()
    frame = datasets.load("digits")
    X = frame[[f"px_{i}" for i in range(64)]].to_numpy(dtype="float32").reshape(-1, 1, 8, 8) / 16.0
    y = frame["digit"].to_numpy(dtype="int64")
    epochs = int(np.clip(config.get("epochs", 5), 1, 30))
    batch_size = int(np.clip(config.get("batch_size", 64), 8, 256))
    learning_rate = float(np.clip(config.get("learning_rate", 0.001), 1e-6, 0.2))
    optimizer_name = str(config.get("optimizer", "adam"))
    dropout = float(np.clip(config.get("dropout", 0.1), 0, 0.7))
    seed = int(np.clip(config.get("seed", 42), 0, 2**31 - 1))
    augmentation = str(config.get("augmentation", "none"))
    device, warnings = _device(str(config.get("device", "auto")))
    _seed_everything(seed)

    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=y)
    augmented_train = _augment_digits(X[train_idx], augmentation, seed + 1)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(augmented_train), torch.from_numpy(y[train_idx])),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = DigitCNN(dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = _optimizer(optimizer_name, model.parameters(), learning_rate, 0.0)
    history = _history_template()
    for epoch in range(1, epochs + 1):
        _append_history(
            history,
            epoch,
            _epoch(model, train_loader, loss_fn, device, optimizer),
            _epoch(model, val_loader, loss_fn, device, None),
        )

    model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(X[val_idx]).to(device)
        logits = model(val_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predicted = probabilities.argmax(axis=1)
        # Feature maps are illustrative outputs from the first learned convolution,
        # not a claim to explain every internal property of the CNN.
        first_conv = model.features[0](val_tensor[:1]).cpu().numpy()[0]
    matrix = confusion_matrix(y[val_idx], predicted, labels=list(range(10))).tolist()
    correct_rows, mistake_rows = [], []
    for local_index, original_index in enumerate(val_idx):
        row = {
            "image": X[original_index, 0].round(4).tolist(),
            "actual": int(y[original_index]),
            "predicted": int(predicted[local_index]),
            "confidence": round(float(probabilities[local_index, predicted[local_index]]), 4),
        }
        (correct_rows if row["actual"] == row["predicted"] else mistake_rows).append(row)
        if len(correct_rows) >= 8 and len(mistake_rows) >= 16:
            break
    duration = time.perf_counter() - started
    resolved = {
        "dataset": "digits",
        "architecture": "Conv2D(1→8) → ReLU → MaxPool → Conv2D(8→16) → ReLU → MaxPool → Linear(64→10)",
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "dropout": dropout,
        "augmentation": augmentation,
        "seed": seed,
        "device_requested": str(config.get("device", "auto")),
    }
    checkpoint = _save_checkpoint(
        model,
        checkpoint_path,
        {"engine": "pytorch", "task": "cnn-digits", "classes": list(range(10)), "config": resolved},
    )
    return {
        "engine": "pytorch",
        "task": "image_classification",
        "dataset": {"id": "digits", "source": "scikit-learn bundled optdigits", "image_shape": [1, 8, 8]},
        "device": str(device),
        "torch_version": str(torch.__version__),
        "duration_seconds": round(duration, 4),
        "seed": seed,
        "config": resolved,
        "parameters": sum(p.numel() for p in model.parameters()),
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "history": history,
        "final": {
            "train_loss": history["train_loss"][-1],
            "validation_loss": history["val_loss"][-1],
            "train_accuracy": history["train_acc"][-1],
            "validation_accuracy": history["val_acc"][-1],
        },
        "confusion_matrix": {"labels": [str(i) for i in range(10)], "matrix": matrix},
        "correct_predictions": correct_rows[:8],
        "misclassified": mistake_rows[:16],
        "augmentation_preview": {
            "mode": augmentation,
            "original": X[train_idx[0], 0].round(4).tolist(),
            "transformed": augmented_train[0, 0].round(4).tolist(),
            "note": "Preview uses the first training image. Augmentation is applied to training data only, never validation data.",
        },
        "feature_maps": [channel.round(4).tolist() for channel in first_conv[:4]],
        "feature_map_note": "Activations from four channels of the first learned convolution for one validation image.",
        "warnings": warnings,
        "checkpoint": checkpoint,
    }
