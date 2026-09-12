from pathlib import Path

import joblib
import numpy as np


class Classifier:
    """
    CodaBench inference wrapper for the mental-health latent probe.

    The trained probe is loaded once during initialization.
    CodaBench supplies latent embeddings to predict().
    """

    def __init__(self):
        model_path = Path(__file__).resolve().parent / "trained_probe.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing trained probe: {model_path}"
            )

        self.model = joblib.load(model_path)

    def predict(self, X):
        X = np.asarray(X)

        predictions = self.model.predict(X)

        return np.asarray(predictions, dtype=int)
