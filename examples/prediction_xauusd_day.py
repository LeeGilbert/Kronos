# -*- coding: utf-8 -*-
"""
prediction_xauusd_day.py

Description:
    Predicts future DAILY K-line (1D) data for XAUUSD (gold spot) using the
    pre-trained Kronos model on a local CSV.

    Why this script exists
    ----------------------
    The XAUUSD CSV has volume=0 and amount=0 for every row (gold spot has no
    exchange volume). In KronosPredictor.predict those all-zero channels get
    normalized by (std + 1e-5) and come back as ~1e-7, which is harmless to
    price but meaningless. We therefore DROP the volume/amount columns and let
    the predictor auto-fill them with zeros, so the model focuses on OHLC.

    The base Kronos model is intraday-oriented and was NOT trained on gold, so
    out-of-the-box daily forecasts are weak (it tends to extrapolate the recent
    trend). For production-quality daily gold forecasts, fine-tune first with
    `finetune_csv/configs/config_xauusd_1d.yaml` and point MODEL_PRETRAINED /
    TOKENIZER_PRETRAINED below at the fine-tuned checkpoints.

Usage:
    python prediction_xauusd_day.py
    python prediction_xauusd_day.py --csv data/XAUUSD_1d_10y.csv --lookback 512 --pred-len 30

Output:
    - <repo_root>/outputs/pred_XAUUSD_1d_data.csv
    - <repo_root>/outputs/pred_XAUUSD_1d_chart.png
"""

import os
import argparse
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # allow headless / no-display runs
import matplotlib.pyplot as plt

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

# Resolve paths relative to the repo root (parent of examples/) so the script
# works regardless of the current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(REPO_ROOT, "outputs")
os.makedirs(save_dir, exist_ok=True)

# Settings
TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"
# Auto-pick a device: CUDA > MPS (Apple Silicon) > CPU
import torch
if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

MAX_CONTEXT = 512   # Kronos-small / Kronos-base max context
LOOKBACK = 512      # daily bars of history (keep <= MAX_CONTEXT)
PRED_LEN = 30       # ~1 month of daily forecast
T = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 8    # average over N sampled paths for a smoother mean


def load_data(csv_path: str) -> pd.DataFrame:
    print(f"📥 Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    # Drop all-zero volume/amount (gold spot has no exchange volume).
    # KronosPredictor.predict auto-fills missing volume/amount with zeros.
    for col in ("volume", "amount"):
        if col in df.columns and (df[col] == 0).all():
            print(f"⚠️  Column '{col}' is all zeros -> dropping it (predictor will re-add zeros).")
            df = df.drop(columns=[col])

    print(f"✅ Loaded {len(df)} rows, range: {df['timestamps'].min().date()} ~ {df['timestamps'].max().date()}")
    print(df[["timestamps", "open", "high", "low", "close"]].tail(3).to_string(index=False))
    return df


def prepare_inputs(df: pd.DataFrame, lookback: int, pred_len: int):
    """Build history window and future timestamps.

    If the file already contains `pred_len` rows after the history window we
    reuse their real timestamps (useful for in-sample evaluation). Otherwise we
    project forward with a business-day range (true out-of-sample forecast).
    """
    x_df = df.iloc[-lookback:][["open", "high", "low", "close"]]
    x_timestamp = df.iloc[-lookback:]["timestamps"]

    tail = df.iloc[-lookback + pred_len:]
    if len(tail) >= pred_len:
        y_timestamp = tail["timestamps"].head(pred_len)
    else:
        last = df["timestamps"].iloc[-1]
        y_timestamp = pd.bdate_range(start=last + pd.Timedelta(days=1), periods=pred_len)

    return x_df, pd.Series(x_timestamp), pd.Series(y_timestamp)


def plot_result(df_hist: pd.DataFrame, pred_df: pd.DataFrame, title: str, out_path: str):
    plt.figure(figsize=(12, 6))
    plt.plot(df_hist["timestamps"], df_hist["close"], label="Historical", color="blue")
    plt.plot(pred_df.index, pred_df["close"], label="Predicted", color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Chart saved: {out_path}")


def predict_future(csv_path: str, lookback: int, pred_len: int):
    print(f"🚀 Loading Kronos tokenizer:{TOKENIZER_PRETRAINED} model:{MODEL_PRETRAINED} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PRETRAINED)
    model = Kronos.from_pretrained(MODEL_PRETRAINED)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)

    df = load_data(csv_path)
    x_df, x_timestamp, y_timestamp = prepare_inputs(df, lookback, pred_len)

    print(f"🔮 Generating {pred_len}-day forecast (lookback={lookback}, device={DEVICE}) ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=T,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
    )

    # The predictor returns volume/amount columns (zeros here); keep OHLC + those.
    pred_df = pred_df.reset_index().rename(columns={"timestamps": "date"})
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    print("Forecasted Data Head:")
    print(pred_df[["date", "open", "high", "low", "close"]].head().to_string(index=False))

    # Merge historical + predicted and save
    hist_cols = ["timestamps", "open", "high", "low", "close"]
    df_out = pd.concat(
        [
            df[hist_cols].rename(columns={"timestamps": "date"}),
            pred_df[["date", "open", "high", "low", "close", "volume", "amount"]],
        ],
        ignore_index=True,
    )
    out_file = os.path.join(save_dir, "pred_XAUUSD_1d_data.csv")
    df_out.to_csv(out_file, index=False)
    print(f"✅ Prediction saved: {out_file}")

    plot_result(df, pred_df, "Kronos Prediction for XAUUSD (1D)",
                os.path.join(save_dir, "pred_XAUUSD_1d_chart.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos XAUUSD daily prediction")
    default_csv = os.path.join(REPO_ROOT, "data", "XAUUSD_1d_5y.csv")
    parser.add_argument("--csv", default=default_csv, help="Input CSV path")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="History window (<= max_context)")
    parser.add_argument("--pred-len", type=int, default=PRED_LEN, help="Number of days to forecast")
    args = parser.parse_args()

    predict_future(csv_path=args.csv, lookback=args.lookback, pred_len=args.pred_len)
