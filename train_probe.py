"""
Train linear probes for the Gemma latent representations.

Official task:
    0 = normal
    1 = mental-health distress

Official evaluation:
    Gemma-2-2B layer 23 embeddings
"""

from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def load_embeddings(path):
    """
    Load an embedding dataset.

    This function will be finalized once we inspect the
    competition's actual public_data file format.
    """
    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)

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
