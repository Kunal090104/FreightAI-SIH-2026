"""
models.py
=========
Small model classes that don't come pre-built from a library:

* :class:`MovingAverageBaseline` — the naive baseline required alongside the
  ML models, so every learned model has to actually beat "just use the
  trailing 7-day average" to be worth deploying.
* :class:`LSTMTabularRegressor` — a thin, joblib-picklable wrapper around a
  Keras LSTM, only ever instantiated by ``train.py`` when TensorFlow is
  installed AND there's enough training history to justify it.

Both classes expose the same minimal ``fit(X, y)`` / ``predict(X)``
interface as scikit-learn estimators so ``train.py`` can treat every
candidate model uniformly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class MovingAverageBaseline:
    """Naive time-series baseline: predicts the next value as the trailing
    N-day rolling mean (default: the ``rolling_mean_7`` feature). No
    fitting is required — it's the bar every real model must clear.
    """

    def __init__(self, feature_name: str = "rolling_mean_7"):
        self.feature_name = feature_name

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MovingAverageBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_name not in X.columns:
            raise ValueError(f"'{self.feature_name}' not found in input features.")
        return X[self.feature_name].to_numpy(dtype=float)


if TENSORFLOW_AVAILABLE:

    class LSTMTabularRegressor:
        """A small single-layer LSTM over the engineered feature vector.

        Each row of tabular features is treated as a length-1 "sequence" of
        ``n_features`` values — this keeps the input contract identical to
        the other models (one row = one feature vector) while still letting
        an LSTM layer learn nonlinear feature interactions. It is only ever
        constructed by ``train.py`` when TensorFlow is available AND the
        training set is large enough (see ``train.MIN_LSTM_TRAIN_SAMPLES``);
        it is never used "for appearance" on too little data.
        """

        def __init__(self, n_features: int, epochs: int = 30, batch_size: int = 32):
            self.n_features = n_features
            self.epochs = epochs
            self.batch_size = batch_size
            self._feature_means: Optional[np.ndarray] = None
            self._feature_stds: Optional[np.ndarray] = None
            self.model = self._build_model()

        def _build_model(self) -> "keras.Model":
            model = keras.Sequential([
                keras.layers.Input(shape=(1, self.n_features)),
                keras.layers.LSTM(32, activation="tanh"),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            return model

        def _scale(self, X: np.ndarray) -> np.ndarray:
            return (X - self._feature_means) / self._feature_stds

        def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMTabularRegressor":
            X_arr = X.to_numpy(dtype=float)
            self._feature_means = X_arr.mean(axis=0)
            self._feature_stds = X_arr.std(axis=0)
            self._feature_stds[self._feature_stds == 0] = 1.0
            X_scaled = self._scale(X_arr).reshape(-1, 1, self.n_features)
            y_arr = np.asarray(y, dtype=float)
            self.model.fit(
                X_scaled, y_arr,
                epochs=self.epochs, batch_size=self.batch_size, verbose=0,
            )
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            X_arr = X.to_numpy(dtype=float)
            X_scaled = self._scale(X_arr).reshape(-1, 1, self.n_features)
            return self.model.predict(X_scaled, verbose=0).reshape(-1)

        # --- joblib/pickle support: Keras models aren't picklable directly,
        # so serialize to a temp .keras file and store its bytes instead. ---
        def __getstate__(self):
            state = self.__dict__.copy()
            model = state.pop("model")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.keras"
                model.save(path)
                state["_model_bytes"] = path.read_bytes()
            return state

        def __setstate__(self, state):
            model_bytes = state.pop("_model_bytes")
            self.__dict__.update(state)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.keras"
                path.write_bytes(model_bytes)
                self.model = keras.models.load_model(path)

else:
    LSTMTabularRegressor = None  # type: ignore
