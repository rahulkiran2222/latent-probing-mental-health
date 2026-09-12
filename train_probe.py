from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

MODEL_NAME = "google/gemma-2-2b"

# Official Project 2 representation
TARGET_LAYER = 23
MAX_LENGTH = 64

# Reproducibility
SEED = 42

# Dataset used for local probe training
DATASET_NAME = "ourafla/Mental-Health_Text-Classification_Dataset"
DATASET_FILE = "mental_heath_unbanlanced.csv"

# Files
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "trained_probe.joblib"
REPORT_PATH = BASE_DIR / "probe_report.json"

TRAIN_EMBEDDINGS_PATH = BASE_DIR / "train_layer23_embeddings.npz"
VAL_EMBEDDINGS_PATH = BASE_DIR / "val_layer23_embeddings.npz"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD DATASET
# ============================================================

def load_training_data():
    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)

    dataset = load_dataset(
        DATASET_NAME,
        data_files=DATASET_FILE
    )

    # Get the first available split
    if "train" in dataset:
        df = dataset["train"].to_pandas()
    else:
        split_name = list(dataset.keys())[0]
        df = dataset[split_name].to_pandas()

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # Find text column
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

        string_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        if not string_columns:
            raise ValueError(
                "Could not identify a text column."
            )

        text_column = string_columns[0]

    # --------------------------------------------------------
    # Find label column
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

        # Try the last column as fallback
        label_column = df.columns[-1]

    print(f"Text column : {text_column}")
    print(f"Label column: {label_column}")

    df = df[
        [text_column, label_column]
    ].copy()

    df.columns = [
        "text",
        "label"
    ]

    # Remove missing values
    df = df.dropna()

    df["text"] = df["text"].astype(str)

    # Remove empty strings
    df = df[
        df["text"].str.strip().str.len() > 0
    ]

    # --------------------------------------------------------
    # Label conversion
    #
    # Project 2:
    #
    # 0 = normal
    # 1 = mental health distress
    #
    # The source dataset contains a Normal class and
    # mental-health-related classes.
    # --------------------------------------------------------

    def convert_label(value):

        value_string = str(value).strip().lower()

        # Explicit normal labels
        normal_labels = {
            "normal",
            "normal text",
            "non-mental-health",
            "non mental health",
            "nonmentalhealth",
            "non mental-health",
            "healthy",
            "none"
        }

        if value_string in normal_labels:
            return 0

        # Explicit binary labels
        if value_string in {
            "0",
            "0.0"
        }:
            return 0

        if value_string in {
            "1",
            "1.0"
        }:
            return 1

        # All other mental-health categories
        # are treated as distress.
        return 1

    df["label"] = df["label"].apply(convert_label)

    print("\nLabel distribution:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nFinal usable examples: {len(df)}"
    )

    return df


# ============================================================
# LOAD GEMMA
# ============================================================

def load_gemma():

    print("\n" + "=" * 70)
    print("LOADING GEMMA 2 2B")
    print("=" * 70)

    from transformers import (
        AutoModel,
        AutoTokenizer
    )

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2
            ),
            "GB"
        )

        dtype = torch.float16

    else:

        device = torch.device("cpu")

        print("WARNING: CUDA is not available.")
        print("CPU inference will be very slow.")

        dtype = torch.float32

    print("Device:", device)
    print("Model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # Gemma may not define a padding token.
    if tokenizer.pad_token is None:

        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype
    )

    model.to(device)
    model.eval()

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    print(
        "Hidden size:",
        model.config.hidden_size
    )

    print(
        "Number of hidden states expected:",
        model.config.num_hidden_layers + 1
    )

    print(
        "Target hidden state:",
        TARGET_LAYER
    )

    return tokenizer, model, device


# ============================================================
# EXTRACT GEMMA LAYER 23 EMBEDDINGS
# ============================================================

def extract_embeddings(
    texts,
    tokenizer,
    model,
    device,
    cache_path,
    batch_size=8
):

    # --------------------------------------------------------
    # Use cached embeddings if available
    # --------------------------------------------------------

    if cache_path.exists():

        print(
            f"\nLoading cached embeddings:"
            f"\n{cache_path}"
        )

        data = np.load(cache_path)

        embeddings = data["embeddings"]

        print(
            "Cached shape:",
            embeddings.shape
        )

        return embeddings

    print("\n" + "=" * 70)
    print("EXTRACTING GEMMA LAYER 23 EMBEDDINGS")
    print("=" * 70)

    print(
        "Number of texts:",
        len(texts)
    )

    print(
        "Layer:",
        TARGET_LAYER
    )

    print(
        "Maximum tokens:",
        MAX_LENGTH
    )

    print(
        "Pooling: mean over valid tokens"
    )

    print(
        "Batch size:",
        batch_size
    )

    hidden_size = model.config.hidden_size

    embeddings = np.zeros(
        (
            len(texts),
            hidden_size
        ),
        dtype=np.float32
    )

    with torch.no_grad():

        for start in range(
            0,
            len(texts),
            batch_size
        ):

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
            # IMPORTANT
            #
            # Project 2 uses Layer 23.
            #
            # hidden_states[23] is the representation used
            # here, following the competition specification.
            # ------------------------------------------------

            hidden = outputs.hidden_states[
                TARGET_LAYER
            ]

            # ------------------------------------------------
            # Mean pooling over VALID tokens only
            # ------------------------------------------------

            attention_mask = (
                encoded["attention_mask"]
            )

            mask = attention_mask.unsqueeze(
                -1
            ).to(
                hidden.dtype
            )

            masked_hidden = (
                hidden * mask
            )

            summed = masked_hidden.sum(
                dim=1
            )

            token_counts = mask.sum(
                dim=1
            ).clamp(
                min=1
            )

            pooled = (
                summed / token_counts
            )

            pooled = (
                pooled
                .float()
                .cpu()
                .numpy()
            )

            embeddings[
                start:end
            ] = pooled

            processed = end
            percent = (
                100.0
                * processed
                / len(texts)
            )

            print(
                f"Progress: "
                f"{processed}/{len(texts)} "
                f"({percent:.1f}%)"
            )

    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    np.savez_compressed(
        cache_path,
        embeddings=embeddings
    )

    print(
        "\nSaved embeddings:",
        cache_path
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    return embeddings


# ============================================================
# TRAIN CANDIDATE PROBES
# ============================================================

def train_candidates(
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\n" + "=" * 70)
    print("TRAINING LINEAR PROBES")
    print("=" * 70)

    candidates = []

    # --------------------------------------------------------
    # Linear SVM candidates
    # --------------------------------------------------------

    svm_c_values = [
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0
    ]

    for C in svm_c_values:

        for class_weight in [
            None,
            "balanced"
        ]:

            name = (
                f"LinearSVC "
                f"C={C} "
                f"class_weight={class_weight}"
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
                (
                    name,
                    model
                )
            )

    # --------------------------------------------------------
    # Logistic Regression candidates
    # --------------------------------------------------------

    logistic_c_values = [
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0
    ]

    for C in logistic_c_values:

        name = (
            f"LogisticRegression C={C}"
        )

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
                    solver="liblinear",
                    class_weight=None,
                    random_state=SEED
                )
            )
        ])

        candidates.append(
            (
                name,
                model
            )
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    best_model = None
    best_name = None
    best_accuracy = -1.0

    for name, model in candidates:

        print(
            f"\nTraining: {name}"
        )

        try:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_val
            )

            accuracy = accuracy_score(
                y_val,
                predictions
            )

            print(
                f"Validation accuracy: "
                f"{accuracy:.5f}"
            )

            result = {
                "model": name,
                "accuracy": float(
                    accuracy
                )
            }

            results.append(result)

            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_model = model
                best_name = name

        except Exception as error:

            print(
                "FAILED:",
                name
            )

            print(
                "Reason:",
                error
            )

    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    results.sort(
        key=lambda item: item["accuracy"],
        reverse=True
    )

    print("\n" + "=" * 70)
    print("TOP VALIDATION RESULTS")
    print("=" * 70)

    for result in results[:10]:

        print(
            f"{result['accuracy']:.5f}"
            f"  ->  "
            f"{result['model']}"
        )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        "Model:",
        best_name
    )

    print(
        "Validation accuracy:",
        f"{best_accuracy:.5f}"
    )

    return (
        best_model,
        best_name,
        best_accuracy,
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PROJECT 2"
    )
    print(
        "LATENT PROBING FOR MENTAL HEALTH "
        "SENTIMENT CLASSIFICATION"
    )
    print("=" * 70)

    print()
    print(
        "Model: Gemma 2 2B"
    )
    print(
        "Official representation: Layer 23"
    )
    print(
        "Pooling: mean pooling"
    )
    print(
        "Maximum tokens: 64"
    )
    print(
        "Target: 0 = normal, "
        "1 = mental-health distress"
    )
    print()

    # ========================================================
    # 1. Load data
    # ========================================================

    df = load_training_data()

    texts = df[
        "text"
    ].tolist()

    labels = df[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    # ========================================================
    # 2. Train/validation split
    # ========================================================

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

    print("\n" + "=" * 70)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    print(
        "Training examples:",
        len(train_texts)
    )

    print(
        "Validation examples:",
        len(val_texts)
    )

    print(
        "Training label distribution:",
        np.bincount(y_train)
    )

    print(
        "Validation label distribution:",
        np.bincount(y_val)
    )

    # ========================================================
    # 3. Load Gemma
    # ========================================================

    tokenizer, model, device = load_gemma()

    # ========================================================
    # 4. Extract training embeddings
    # ========================================================

    batch_size = 8 if torch.cuda.is_available() else 2

    X_train = extract_embeddings(
        texts=train_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        cache_path=TRAIN_EMBEDDINGS_PATH,
        batch_size=batch_size
    )

    # ========================================================
    # 5. Extract validation embeddings
    # ========================================================

    X_val = extract_embeddings(
        texts=val_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        cache_path=VAL_EMBEDDINGS_PATH,
        batch_size=batch_size
    )

    # ========================================================
    # 6. Train candidate probes
    # ========================================================

    (
        best_model,
        best_name,
        best_accuracy,
        results
    ) = train_candidates(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    # ========================================================
    # 7. Retrain best probe on ALL available data
    # ========================================================

    print("\n" + "=" * 70)
    print("RETRAINING BEST PROBE ON ALL DATA")
    print("=" * 70)

    X_all = np.concatenate(
        [
            X_train,
            X_val
        ],
        axis=0
    )

    y_all = np.concatenate(
        [
            y_train,
            y_val
        ],
        axis=0
    )

    print(
        "Final training shape:",
        X_all.shape
    )

    best_model.fit(
        X_all,
        y_all
    )

    # ========================================================
    # 8. Save trained probe
    # ========================================================

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print(
        "\nSaved trained probe:"
    )

    print(
        MODEL_PATH
    )

    # ========================================================
    # 9. Save experiment report
    # ========================================================

    report = {
        "project": (
            "Latent Probing for Mental Health "
            "Sentiment Classification"
        ),
        "model": MODEL_NAME,
        "layer": TARGET_LAYER,
        "pooling": (
            "mean_pool_all_valid_tokens"
        ),
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "dataset": DATASET_NAME,
        "dataset_file": DATASET_FILE,
        "total_examples": int(
            len(df)
        ),
        "train_examples": int(
            len(train_texts)
        ),
        "validation_examples": int(
            len(val_texts)
        ),
        "best_model": best_name,
        "validation_accuracy": float(
            best_accuracy
        ),
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
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print(
        "Saved experiment report:"
    )

    print(
        REPORT_PATH
    )

    # ========================================================
    # 10. Final summary
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model:               {MODEL_NAME}"
    )

    print(
        f"Layer:               {TARGET_LAYER}"
    )

    print(
        f"Pooling:             mean"
    )

    print(
        f"Maximum tokens:      {MAX_LENGTH}"
    )

    print(
        f"Examples:            {len(df)}"
    )

    print(
        f"Best probe:          {best_name}"
    )

    print(
        f"Validation accuracy: "
        f"{best_accuracy:.5f}"
    )

    print()
    print(
        "Generated files:"
    )

    print(
        f"  {MODEL_PATH.name}"
    )

    print(
        f"  {REPORT_PATH.name}"
    )

    print()
    print(
        "Embedding cache files:"
    )

    print(
        f"  {TRAIN_EMBEDDINGS_PATH.name}"
    )

    print(
        f"  {VAL_EMBEDDINGS_PATH.name}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
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
