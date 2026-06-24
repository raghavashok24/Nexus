"""Data loading utilities for the EWS NexusAI pipeline.

The original notebook loaded data via a Google Colab file-upload widget
(`google.colab.files.upload()`), which only works inside Colab. This module
replaces that with a plain local-path loader so the pipeline runs anywhere.
"""

from __future__ import annotations

import pandas as pd


def load_bank_panel(csv_path: str) -> pd.DataFrame:
    """Load the quarterly bank panel dataset from a local CSV path.

    Parameters
    ----------
    csv_path:
        Path to the CSV file containing the bank panel data. Expected to
        include a ``probabilistic_risk_score`` target column, a ``Quarter``
        column for the time-based split, and the accounting / macro /
        non-accounting feature columns described in the README.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified panel data.
    """
    return pd.read_csv(csv_path)
