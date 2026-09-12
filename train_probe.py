from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import joblib

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "google/gemma-2-2b"

# Competition specification
TARGET_LAYER = 23
MAX_LENGTH = 64

# Reproducibility
SEED = 42

# Files produced by this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_probe.joblib"
REPORT_PATH = BASE_DIR / "probe_report.json"


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Dataset
# ============================================================

DATASET_NAME = "ourafla/Mental-Health_Text-Classification_Dataset"
DATASET_FILE = "mental_heath_unbanlanced.csv"


def load_training_data():
    print("=" * 70)
    print("Loading training dataset")
    print("=" * 70)

    dataset = load_dataset(
        DATASET_NAME,
        data_files=DATASET_FILE
    )

    # Handle common dataset layouts
    if "train" in dataset:
        df = dataset["train"].to_pandas()
    else:
        first_split = list(dataset.keys())[0]
        df = dataset[first_split].to_pandas()

    print(f"Dataset shape: {df.shape}")
    print("Columns:", list(df.columns))

    # --------------------------------------------------------
    # Detect text column
    # --------------------------------------------------------

    text_candidates = [
        "text",
        "Text",
        "sentence",
        "Sentence",
        "content",
        "Content",
        "statement",
        "Statement"
    ]

    text_column = None

    for column in text_candidates:
        if column in df.columns:
            text_column = column
            break

    if text_column is None:
        # Fall back to first object/string column
        object_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        if not object_columns:
            raise ValueError(
                "Could not find a text column in the dataset."
            )

        text_column = object_columns[0]

    # --------------------------------------------------------
    # Detect label column
    # --------------------------------------------------------

    label_candidates = [
        "label",
        "Label",
        "status",
        "Status",
        "class",
        "Class",
        "category",
        "Category"
    ]

    label_column = None

    for column in label_candidates:
        if column in df.columns:
            label_column = column
            break

    if label_column is None:
        # Use the last column as a fallback
        label_column = df.columns[-1]

    print(f"Text column : {text_column}")
    print(f"Label column: {label_column}")

    df = df[[text_column, label_column]].copy()
    df.columns = ["text", "label"]

    df = df.dropna()
    df["text"] = df["text"].astype(str)

    # --------------------------------------------------------
    # Convert labels
    #
    # Competition:
    #   0 = normal
    #   1 = mental health distress
    # --------------------------------------------------------

    def convert_label(value):
        value_str = str(value).strip().lower()

        # Already binary
        if value_str in {"0", "0.0"}:
            return 0

        if value_str in {"1", "1.0"}:
            return 1

        # Normal examples
        normal_values = {
            "normal",
            "normal text",
            "non-mental-health",
            "non mental health",
            "nonmentalhealth"
        }

        if value_str in normal_values:
            return 0

        # Everything else is treated as distress,
        # matching the approach used previously.
        return 1

    df["label"] = df["label"].apply(convert_label)

    df = df[df["text"].str.len() > 0]

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    print(f"\nFinal training examples: {len(df)}")

    return df


# ============================================================
# Gemma embedding extraction
# ============================================================

def load_gemma():
    print("\n" + "=" * 70)
    print("Loading Gemma")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available()
        else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if not torch.cuda.is_available():
        model = model.to(device)

    model.eval()

    return tokenizer, model


def extract_embeddings(texts, tokenizer, model):
    print("\n" + "=" * 70)
    print("Extracting Gemma Layer 23 embeddings")
    print("=" * 70)

    device = next(model.parameters()).device

    # Gemma 2 hidden size
    hidden_size = model.config.hidden_size

    embeddings = np.zeros(
        (len(texts), hidden_size),
        dtype=np.float32
    )

    batch_size = 16 if torch.cuda.is_available() else 2

    print(f"Number of texts : {len(texts)}")
    print(f"Hidden size     : {hidden_size}")
    print(f"Target layer    : {TARGET_LAYER}")
    print(f"Max tokens      : {MAX_LENGTH}")
    print(f"Batch size      : {batch_size}")

    with torch.no_grad():

        for start in range(0, len(texts), batch_size):

            end = min(
                start + batch_size,
                len(texts)
            )

            batch_texts = texts[start:end]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded,
                output_hidden_states=True
            )

            # ------------------------------------------------
            # Competition specification:
            # Layer 23
            # Mean across ALL VALID TOKENS
            # ------------------------------------------------

            hidden = outputs.hidden_states[TARGET_LAYER]

            attention_mask = encoded["attention_mask"]

            mask = attention_mask.unsqueeze(-1).to(
                hidden.dtype
            )

            summed = (hidden * mask).sum(dim=1)

            counts = mask.sum(dim=1).clamp(
                min=1
            )

            pooled = summed / counts

            embeddings[start:end] = (
                pooled
                .float()
                .cpu()
                .numpy()
            )

            if start % (batch_size * 20) == 0:
                progress = 100 * end / len(texts)
                print(
                    f"Progress: {end}/{len(texts)} "
                    f"({progress:.1f}%)"
                )

    print("Embedding extraction complete.")
    print("Embedding shape:", embeddings.shape)

    return embeddings


# ============================================================
# Train candidate probes
# ============================================================

def train_candidates(X_train, y_train, X_val, y_val):

    print("\n" + "=" * 70)
    print("Training linear probes")
    print("=" * 70)

    candidates = []

    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    for C in [
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0
    ]:

        for class_weight in [
            None,
            "balanced"
        ]:

            name = (
                f"LinearSVC_C={C}_"
                f"weight={class_weight}"
            )

            model = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    LinearSVC(
                        C=C,
                        class_weight=class_weight,
                        max_iter=20000,
                        dual="auto",
                        random_state=SEED
                    )
                )
            ])

            candidates.append(
                (name, model)
            )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    for C in [
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0
    ]:

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=5000,
                    class_weight=None,
                    solver="liblinear",
                    random_state=SEED
                )
            )
        ])

        candidates.append(
            (
                f"LogisticRegression_C={C}",
                model
            )
        )

    results = []

    best_model = None
    best_name = None
    best_accuracy = -1

    for name, model in candidates:

        print(f"\nTraining: {name}")

        try:
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            accuracy = accuracy_score(
                y_val,
                predictions
            )

            print(
                f"Validation accuracy: "
                f"{accuracy:.5f}"
            )

            results.append({
                "model": name,
                "accuracy": float(accuracy)
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name

        except Exception as e:

            print(
                f"FAILED: {name}"
            )
            print(e)

    results.sort(
        key=lambda x: x["accuracy"],
        reverse=True
    )

    print("\n" + "=" * 70)
    print("TOP VALIDATION RESULTS")
    print("=" * 70)

    for result in results[:10]:

        print(
            f"{result['accuracy']:.5f} "
            f"-> {result['model']}"
        )

    print("\nBEST MODEL:")
    print(best_name)
    print(f"Validation accuracy: {best_accuracy:.5f}")

    return (
        best_model,
        best_name,
        best_accuracy,
        results
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PROJECT 2 — LATENT PROBING")
    print("Mental Health Sentiment Classification")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_training_data()

    texts = df["text"].tolist()
    labels = df["label"].to_numpy(dtype=np.int64)

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    (
        train_texts,
        val_texts,
        y_train,
        y_val
    ) = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )

    print("\nTrain examples:", len(train_texts))
    print("Validation examples:", len(val_texts))

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    tokenizer, model = load_gemma()

    # --------------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------------

    X_train = extract_embeddings(
        train_texts,
        tokenizer,
        model
    )

    X_val = extract_embeddings(
        val_texts,
        tokenizer,
        model
    )

    # --------------------------------------------------------
    # Train probes
    # --------------------------------------------------------

    (
        best_model,
        best_name,
        best_accuracy,
        results
    ) = train_candidates(
        X_train,
        y_train,
        X_val,
        y_val
    )

    # --------------------------------------------------------
    # Retrain BEST model on ALL available data
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Retraining best probe on ALL training data")
    print("=" * 70)

    X_all = np.vstack([
        X_train,
        X_val
    ])

    y_all = np.concatenate([
        y_train,
        y_val
    ])

    best_model.fit(
        X_all,
        y_all
    )

    # --------------------------------------------------------
    # Save classifier
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print(
        f"\nSaved trained probe to:\n"
        f"{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {
        "project": (
            "Latent Probing for Mental Health "
            "Sentiment Classification"
        ),
        "model": MODEL_NAME,
        "layer": TARGET_LAYER,
        "pooling": "mean_pool_all_valid_tokens",
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "training_examples": len(df),
        "train_examples": len(train_texts),
        "validation_examples": len(val_texts),
        "best_model": best_name,
        "validation_accuracy": float(best_accuracy),
        "label_mapping": {
            "0": "normal",
            "1": "mental_health_distress"
        },
        "results": results
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"Saved report to:\n"
        f"{REPORT_PATH}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"Layer:              {TARGET_LAYER}")
    print(f"Pooling:            Mean")
    print(f"Max tokens:         {MAX_LENGTH}")
    print(f"Training examples:  {len(df)}")
    print(f"Best probe:         {best_name}")
    print(f"Validation accuracy:{best_accuracy:.5f}")

    print("\nFiles created:")

    print("  trained_probe.joblib")
    print("  probe_report.json")


if __name__ == "__main__":
    main()        data = np.load(path, allow_pickle=True)

        print("Available keys:")
        for key in data.files:
            print(f"  {key}: {data[key].shape}")

        return data

    raise ValueError(
        f"Unsupported embedding format: {path.suffix}"
    )


def build_candidates():
    """
    Candidate probes to compare on the official embedding data.
    """

    return {
        "linear_svc_c_0.01": LinearSVC(
            C=0.01,
            class_weight=None,
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),

        "linear_svc_c_0.1": LinearSVC(
            C=0.1,
            class_weight=None,
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),

        "linear_svc_c_1": LinearSVC(
            C=1.0,
            class_weight=None,
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),

        "linear_svc_balanced": LinearSVC(
            C=0.1,
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),

        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_candidates(X_train, y_train, X_valid, y_valid):
    results = []

    for name, model in build_candidates().items():

        print(f"\nTraining: {name}")

        model.fit(X_train, y_train)

        predictions = model.predict(X_valid)

        accuracy = accuracy_score(
            y_valid,
            predictions,
        )

        print(f"Validation accuracy: {accuracy:.6f}")

        results.append(
            {
                "name": name,
                "model": model,
                "accuracy": accuracy,
            }
        )

    results.sort(
        key=lambda item: item["accuracy"],
        reverse=True,
    )

    return results


def main():
    print("=" * 70)
    print("MENTAL HEALTH LATENT PROBE")
    print("=" * 70)

    print()
    print("Official representation: Gemma-2-2B Layer 23")
    print("Pooling: mean pooling")
    print("Maximum tokens: 64")
    print("Target: 0 = normal, 1 = mental-health distress")
    print()

    # We will connect this to the actual competition
    # public_data after inspecting its file structure.
    print(
        "Next step: provide the official public embedding "
        "dataset so the loader can be finalized."
    )


if __name__ == "__main__":
    main()
