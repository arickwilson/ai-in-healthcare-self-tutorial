# AI in Healthcare - MIMIC Length-of-Stay Prediction Example

# Self tutorial and example code for loading MIMIC-III data from BigQuery,
# preprocessing, building PyTorch DataLoaders, defining a simple regression
# model for length-of-stay prediction.

# Arick Wilson

from typing import Optional, Tuple, List, Sequence, Dict
import os
import tempfile
import math
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from google.cloud import bigquery
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

SQL_QUERY = r"""
-- BigQuery Standard SQL version of the LoS builder
WITH
icu_core AS (
    SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.icustay_id,
        icu.intime,
        icu.outtime,
        ROUND(SAFE_DIVIDE(TIMESTAMP_DIFF(icu.outtime, icu.intime, SECOND), 3600.0), 2) AS los_hours
    FROM `physionet-data.mimiciii_clinical.icustays` icu
    WHERE icu.outtime IS NOT NULL
),

demographics AS (
    SELECT
        p.subject_id,
        a.hadm_id,
        EXTRACT(YEAR FROM a.admittime) - EXTRACT(YEAR FROM p.dob) AS age,
        IF(p.gender = 'M', 1, 0) AS is_male,
        a.ethnicity,
        a.admission_type,
        a.insurance,
        a.marital_status
    FROM `physionet-data.mimiciii_clinical.patients` p
    JOIN `physionet-data.mimiciii_clinical.admissions` a ON p.subject_id = a.subject_id
),

diagnoses AS (
    SELECT
        di.hadm_id,
        COUNT(DISTINCT di.icd9_code) AS num_diagnoses,
        SUM(CASE WHEN SAFE_CAST(REGEXP_REPLACE(di.icd9_code, '[^0-9]', '') AS INT64) BETWEEN 25000 AND 25099 THEN 1 ELSE 0 END) AS has_diabetes,
        SUM(CASE WHEN SAFE_CAST(REGEXP_REPLACE(di.icd9_code, '[^0-9]', '') AS INT64) BETWEEN 41000 AND 41499 THEN 1 ELSE 0 END) AS has_cardiac
    FROM `physionet-data.mimiciii_clinical.diagnoses_icd` di
    JOIN `physionet-data.mimiciii_clinical.d_icd_diagnoses` d ON di.icd9_code = d.icd9_code
    GROUP BY di.hadm_id
),

lab_summary AS (
    SELECT
        l.hadm_id,
        AVG(IF(itemid = 50983, valuenum, NULL)) AS avg_glucose,
        AVG(IF(itemid = 50868, valuenum, NULL)) AS avg_lactate,
        AVG(IF(itemid = 50912, valuenum, NULL)) AS avg_creatinine,
        COUNT(*) AS lab_count_24h
    FROM `physionet-data.mimiciii_clinical.labevents` l
    JOIN `physionet-data.mimiciii_clinical.admissions` a ON l.hadm_id = a.hadm_id
    WHERE l.charttime BETWEEN a.admittime AND TIMESTAMP_ADD(a.admittime, INTERVAL 24 HOUR)
        AND l.valuenum IS NOT NULL
    GROUP BY l.hadm_id
),

vitals_summary AS (
    SELECT
        c.icustay_id,
        AVG(IF(itemid IN (211, 220045), valuenum, NULL)) AS avg_heart_rate,
        AVG(IF(itemid IN (51, 442, 455, 6701, 220179, 220050), valuenum, NULL)) AS avg_sbp,
        AVG(IF(itemid IN (456, 443, 220180, 220051), valuenum, NULL)) AS avg_dbp,
        AVG(IF(itemid IN (223761, 678), valuenum, NULL)) AS avg_temp,
        COUNT(*) AS vitals_count_24h
    FROM `physionet-data.mimiciii_clinical.chartevents` c
    JOIN `physionet-data.mimiciii_clinical.icustays` i ON c.icustay_id = i.icustay_id
    WHERE c.charttime BETWEEN i.intime AND TIMESTAMP_ADD(i.intime, INTERVAL 24 HOUR)
        AND c.valuenum IS NOT NULL
    GROUP BY c.icustay_id
)

SELECT
    i.icustay_id,
    i.subject_id,
    i.hadm_id,
    i.los_hours,
    d.age,
    d.is_male,
    d.ethnicity,
    d.admission_type,
    d.insurance,
    d.marital_status,
    dx.num_diagnoses,
    dx.has_diabetes,
    dx.has_cardiac,
    l.avg_glucose,
    l.avg_lactate,
    l.avg_creatinine,
    v.avg_heart_rate,
    v.avg_sbp,
    v.avg_dbp,
    v.avg_temp
FROM icu_core i
LEFT JOIN demographics d ON i.subject_id = d.subject_id AND i.hadm_id = d.hadm_id
LEFT JOIN diagnoses dx ON i.hadm_id = dx.hadm_id
LEFT JOIN lab_summary l ON i.hadm_id = l.hadm_id
LEFT JOIN vitals_summary v ON i.icustay_id = v.icustay_id;
"""

# Note: DEFAULT_SQL is a convenience example using the public `physionet-data` dataset.
# Replace it or pass a custom SQL string when creating a `MIMICDataLoader` if you
# have a different schema or want a different feature set.


def fetch_bigquery(sql: str, project: Optional[str] = None, use_bqstorage: bool = False) -> pd.DataFrame:
    """
    Run a BigQuery SQL string and return a pandas DataFrame.
    """
    client = bigquery.Client(project=project)
    job = client.query(sql)
    df = job.to_dataframe(create_bqstorage_client=use_bqstorage)
    return df


def preprocess(df: pd.DataFrame, one_hot: bool = False, numeric_fill: str = "median", max_los_hours: Optional[float] = 720.0) -> pd.DataFrame:
    """
    Basic preprocessing for MIMIC length-of-stay DataFrame.
    """
    df = df.copy()
    # ensure label present
    if "los_hours" not in df.columns:
        raise ValueError("DataFrame must contain 'los_hours' column")

    df = df[df["los_hours"].notna()]
    df = df[df["los_hours"] > 0]

    # filter extreme outliers: drop stays longer than max_los_hours (hours). Default = 720 (30 days)
    if max_los_hours is not None:
        try:
            df = df[df["los_hours"] <= float(max_los_hours)]
        except Exception:
            # if conversion fails, skip outlier filtering
            pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "los_hours"]
    cat_cols = df.select_dtypes(include=[object, "category"]).columns.tolist()

    # fill missing values
    for c in numeric_cols:
        if df[c].isna().any():
            fill = df[c].median() if numeric_fill == "median" else df[c].mean()
            df[c] = df[c].fillna(fill)

    # categorical fill
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna("Unknown")

    # one-hot encode categoricals
    if one_hot and len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    return df


class MIMICDataset:
    """
    PyTorch Dataset wrapping a processed pandas DataFrame.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: Sequence[str], label_col: str = "los_hours"):
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.df = df.reset_index(drop=True)

        X = self.df[self.feature_cols].astype(np.float32).to_numpy()
        y = self.df[self.label_col].astype(np.float32).to_numpy().reshape(-1, 1)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MIMICDataLoader:
    """
    High-level convenience class to produce train/validation DataLoaders.

    This class handles fetching (BigQuery or cached parquet), simple
    preprocessing, a stratified train/validation split, and construction of
    PyTorch Datasets and DataLoaders.
    """
    def __init__(
        self,
        project: Optional[str] = None,
        sql: Optional[str] = None,
        cache_path: Optional[str] = None,
        use_bqstorage: bool = False,
        one_hot: bool = False,
        val_size: float = 0.3,
        random_state: int = 42,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        numeric_fill: str = "median",
        max_los_hours: Optional[float] = 720.0,
    ):
        self.project = project
        self.sql = SQL_QUERY if sql is None else sql
        self.cache_path = cache_path
        self.use_bqstorage = use_bqstorage
        self.one_hot = one_hot
        self.val_size = val_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.numeric_fill = numeric_fill
        self.max_los_hours = max_los_hours

        # load dataframe
        self.df = self.load_dataframe()

        # preprocess (drop NA/invalid and optionally filter extreme LOS outliers)
        self.df = preprocess(
            self.df, one_hot=self.one_hot, numeric_fill=self.numeric_fill, max_los_hours=self.max_los_hours
        )

        # choose feature columns (exclude identifiers and label)
        exclude = {"icustay_id", "subject_id", "hadm_id", "intime", "outtime", "los_hours"}
        self.feature_cols = [c for c in self.df.columns if c not in exclude]

        # split
        self.train_df, self.val_df = self.split(self.df, val_size=self.val_size, random_state=self.random_state)

        # build datasets and loaders
        self.build_loaders()

    def load_dataframe(self) -> pd.DataFrame:
        # If cache exists, load it
        if self.cache_path and os.path.exists(self.cache_path):
            return pd.read_parquet(self.cache_path)

        df = fetch_bigquery(self.sql, project=self.project, use_bqstorage=self.use_bqstorage)

        if self.cache_path:
            try:
                df.to_parquet(self.cache_path)
            except Exception:
                # don't fail on caching problems
                pass

        return df

    def split(self, df: pd.DataFrame, val_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # create stratify bins from los_hours quantiles where possible
        try:
            stratify = pd.qcut(df["los_hours"], q=10, duplicates="drop")
        except Exception:
            stratify = None

        train_df, val_df = train_test_split(df, test_size=val_size, random_state=random_state, stratify=stratify)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def build_loaders(self):
        # DataLoader imported at module level

        self.train_ds = MIMICDataset(self.train_df, self.feature_cols, label_col="los_hours")
        self.val_ds = MIMICDataset(self.val_df, self.feature_cols, label_col="los_hours")

        self.train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def get_feature_names(self) -> List[str]:
        return list(self.feature_cols)


# Simple regression model and training loop using the MIMICDataLoader
class SimpleRegressor(nn.Module):
    """
    Small multi-layer perceptron for scalar regression.
    """
    def __init__(self, input_dim: int, hidden_dims=(256, 256, 128, 64), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out.view(-1)

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = None,
    tensorboard_logdir: Optional[str] = None,
    weight_decay: float = 0.0,
    loss_type: str = "mse",
    early_stopping_patience: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, list]]:
    # device selection: prefer CUDA if available, otherwise CPU
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # optimizer with optional weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # choose loss
    loss_type = (loss_type or "mse").lower()
    if loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif loss_type in ("huber", "smoothl1"):
        loss_fn = nn.SmoothL1Loss()
    elif loss_type in ("mae", "l1"):
        loss_fn = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    best_val = float("inf")
    best_path = None
    wait = 0

    writer = None
    # If a tensorboard directory is provided, create a unique child folder and
    # write run artifacts there to avoid overwriting previous runs.
    if tensorboard_logdir is not None:
        # create a unique child directory for this run under the provided logdir
        # e.g. tensorboard_logdir/<YYYYmmdd-HHMMSS>-<short-uuid>/
        def make_unique_run_dir(base_dir: str) -> str:
            os.makedirs(base_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            uid = uuid.uuid4().hex[:8]
            run_dir = os.path.join(base_dir, f"{ts}-{uid}")
            # attempt to create the directory; if it races, fall back to a uuid-only name
            try:
                os.makedirs(run_dir, exist_ok=False)
            except Exception:
                run_dir = os.path.join(base_dir, uuid.uuid4().hex)
                os.makedirs(run_dir, exist_ok=True)
            return run_dir

    unique_logdir = make_unique_run_dir(tensorboard_logdir)
    writer = SummaryWriter(log_dir=unique_logdir)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            batch_n = xb.size(0)
            total_loss += loss.item() * batch_n
            total_n += batch_n
        train_loss = total_loss / max(total_n, 1)

        model.eval()
        total_loss = 0.0
        total_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).view(-1)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                batch_n = xb.size(0)
                total_loss += loss.item() * batch_n
                total_n += batch_n
        val_loss = total_loss / max(total_n, 1)

        # early stopping / checkpointing
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            try:
                # save best model to a temporary file
                fd, p = tempfile.mkstemp(prefix="best_model_", suffix=".pt")
                os.close(fd)
                torch.save(model.state_dict(), p)
                best_path = p
            except Exception:
                best_path = None
        else:
            wait += 1
            if early_stopping_patience is not None and wait >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {wait} epochs)")
                break

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # log to tensorboard if requested
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            # also log RMSE
            rmse = math.sqrt(val_loss)
            writer.add_scalar("RMSE/val", rmse, epoch)

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass

    # if we saved a best model, load it back
    if best_path is not None:
        try:
            model.load_state_dict(torch.load(best_path, map_location=device))
            try:
                os.remove(best_path)
            except Exception:
                pass
        except Exception:
            pass

    return model, history

if __name__ == "__main__":
    # this is my project ID for the BigQuery MIMIC-III dataset, update as needed
    loader = MIMICDataLoader(project='mimic-iii-473123', cache_path=None, one_hot=True)
    train_dl = loader.train_loader
    val_dl = loader.val_loader
    input_dim = len(loader.get_feature_names())
    model = SimpleRegressor(input_dim)
    trained_model, hist = train_model(model, train_dl, val_dl, epochs=200, tensorboard_logdir="runs")
    print("Training complete.")