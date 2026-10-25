# -*- coding: utf-8 -*-
"""
prediction_xauusd_day.py

Description:
    Predicts future DAILY K-line (1D) data for XAUUSD (gold spot) with Kronos,
    using a Monte Carlo ensemble of sampled paths instead of a single point
    forecast, plus a rolling walk-forward backtest.

    Why this script exists
    ----------------------
    1. The XAUUSD CSV has volume=0 and amount=0 for every row (gold spot has no
       exchange volume). In KronosPredictor.predict those all-zero channels get
       normalized by (std + 1e-5) and come back as ~1e-7, which is harmless to
       price but meaningless. We therefore DROP the volume/amount columns and
       let the predictor auto-fill them with zeros, so the model focuses on OHLC.

    2. The base Kronos model is intraday-oriented and was NOT trained on gold, so
       out-of-the-box daily forecasts are weak (it tends to extrapolate the
       recent trend and under-predict volatility). This script therefore:
         * defaults to the fine-tuned checkpoints in
           finetune_csv/finetuned/XAUUSD_1d_10y/{tokenizer,basemodel}/best_model
           (fall back to the HF pre-trained models with --pretrained, or
           automatically if the fine-tuned dirs are missing);
         * samples SAMPLE_COUNT=32 independent stochastic paths in ONE batched
           auto-regressive loop (KronosPredictor.predict averages paths
           internally and can only return one), with a lower temperature
           (T=0.7) to reduce the downward drift of individual paths;
         * reports the ensemble mean plus 5-95% / 25-75% quantile bands and
           probability metrics (upside/downside probability, VaR/CVaR,
           max drawdown);
         * calibrates the ensemble volatility post-hoc (see calibrate_vol):
           each path's deviation from the ensemble mean is rescaled so the
           per-horizon spread matches the recent realized daily vol
           (sigma_d * sqrt(t)), fixing Kronos's ~2-5x vol compression while
           leaving the mean path (and thus MAPE/bias) untouched; the
           ensemble is also mapped to a normal shape (rank -> Phi^-1
           quantile) so the 25-75% / 5-95% bands contain 50% / 90% of mass
           by construction (disable with --no-vol-shape);
         * runs a rolling walk-forward backtest (5 non-overlapping 30-day
           windows at the tail of the file, which lies inside the fine-tuning
           validation split) tracking MAPE, signed bias, return correlation,
           direction accuracy, realized-vol ratio and prediction-interval
           coverage per window/regime.

    Forward vs in-sample
    --------------------
    * --forward (default): true out-of-sample forecast. The history window is
      the last `lookback` rows and the forecast timestamps are projected
      forward from the last date with a business-day range.
    * --no-forward: in-sample evaluation. The history window ends `pred_len`
      rows before the file end and the "future" timestamps are the file's real
      tail dates, so the forecast can be compared against actuals.

    Usage:
        python examples/prediction_xauusd_day.py \
            --csv data/XAUUSD_1d_10y.csv --lookback 512 --pred-len 30

    Output (in <repo_root>/outputs/):
        - pred_XAUUSD_1d_data.csv      history + mean forecast
        - pred_XAUUSD_1d_chart.png     history + mean + quantile bands
        - backtest_XAUUSD_1d.csv        per-window backtest metrics
        - backtest_XAUUSD_1d_daily.csv  per-day predicted vs actual
        - backtest_XAUUSD_1d_corr.png   predicted vs actual return scatter
        - backtest_XAUUSD_1d_chart.png  per-window actual vs predicted
"""

import os
import argparse
import sys
import statistics

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # allow headless / no-display runs
import matplotlib.pyplot as plt
import torch

# Resolve paths relative to the repo root (parent of examples/) so the script
# works regardless of the current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from model import Kronos, KronosTokenizer, KronosPredictor
from model.kronos import calc_time_stamps, sample_from_logits
save_dir = os.path.join(REPO_ROOT, "outputs")
os.makedirs(save_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Model checkpoints: fine-tuned by default, pre-trained as fallback
# ---------------------------------------------------------------------------
FINETUNED_DIR = os.path.join(REPO_ROOT, "finetune_csv", "finetuned", "XAUUSD_1d_10y")
FINETUNED_TOKENIZER = os.path.join(FINETUNED_DIR, "tokenizer", "best_model")
FINETUNED_MODEL = os.path.join(FINETUNED_DIR, "basemodel", "best_model")
PRETRAINED_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
PRETRAINED_MODEL = "NeoQuasar/Kronos-base"


def resolve_model_paths(use_pretrained: bool):
    """Pick tokenizer/model checkpoints.

    Fine-tuned checkpoints are preferred (they were trained on daily XAUUSD);
    the HF pre-trained models are used with --pretrained, or automatically if
    the fine-tuned directories do not exist yet.
    """
    if use_pretrained:
        return PRETRAINED_TOKENIZER, PRETRAINED_MODEL
    if os.path.isdir(FINETUNED_TOKENIZER) and os.path.isdir(FINETUNED_MODEL):
        return FINETUNED_TOKENIZER, FINETUNED_MODEL
    print(f"⚠️  Fine-tuned checkpoints not found in {FINETUNED_DIR}")
    print("    Falling back to the HF pre-trained models (run finetune_csv/train_sequential.py first).")
    return PRETRAINED_TOKENIZER, PRETRAINED_MODEL


# Auto-pick a device: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

MAX_CONTEXT = 512   # Kronos-small / Kronos-base max context
LOOKBACK = 512      # daily bars of history (keep <= MAX_CONTEXT)
PRED_LEN = 30       # ~1 month of daily forecast
T = 0.7             # lower temperature: tames the downward drift of raw paths
TOP_P = 0.9
SAMPLE_COUNT = 32   # Monte Carlo ensemble size (one batched AR loop)
PRICE_COLS = ["open", "high", "low", "close"]
CLOSE_IDX = 3       # index of 'close' in the model's 6 output channels
PRICE_IDX = (0, 1, 2, 3)  # open, high, low, close in the model's 6 output channels
VOL_LOOKBACK = 30   # history days used to estimate the recent realized vol (vol-targeting)
NORMAL = statistics.NormalDist()  # stdlib inverse normal CDF (scipy is not installed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
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


def prepare_inputs(df: pd.DataFrame, lookback: int, pred_len: int,
                   forward: bool = False, end_offset: int = 0):
    """Build the history window and the future timestamps.

    forward=True (true out-of-sample forecast):
        x = the last `lookback` rows of the file;
        y = a business-day range projected forward from the last date.

    forward=False (in-sample evaluation):
        The history window ends `pred_len + end_offset` rows before the file
        end, and y is the file's REAL dates for the following `pred_len` rows,
        so the forecast can be compared against actuals. `end_offset` slides
        the window backwards (used by the rolling backtest).
    """
    if forward:
        x_df = df.iloc[-lookback:]
        last = df["timestamps"].iloc[-1]
        y_timestamp = pd.bdate_range(start=last + pd.Timedelta(days=1), periods=pred_len)
    else:
        end = len(df) - pred_len - end_offset
        if end - lookback < 0:
            raise ValueError(
                f"Not enough history: need {lookback} rows before the window end "
                f"(end={end}, file has {len(df)} rows). Reduce --lookback or --pred-len."
            )
        x_df = df.iloc[end - lookback:end]
        y_timestamp = df["timestamps"].iloc[end:end + pred_len]

    x_df = x_df[["timestamps"] + PRICE_COLS]
    return x_df, pd.Series(x_df["timestamps"].values), pd.Series(y_timestamp.values)


# ---------------------------------------------------------------------------
# Monte Carlo sampling (batched, un-averaged)
# ---------------------------------------------------------------------------
def _auto_regressive_paths(tokenizer, model, x, x_stamp, y_stamp, max_context,
                           pred_len, clip, T, top_k, top_p, n_paths, verbose=False):
    """Auto-regressive sampling that returns ALL paths un-averaged.

    Mirrors model.kronos.auto_regressive_inference (which averages the
    `sample_count` paths into one) but keeps every sampled path, so a single
    batched loop of `n_paths` yields an ensemble instead of n_paths separate
    batch-1 loops.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = torch.clip(x, -clip, clip)

            device = x.device
            x = x.unsqueeze(1).repeat(1, n_paths, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
            x_stamp = x_stamp.unsqueeze(1).repeat(1, n_paths, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
            y_stamp = y_stamp.unsqueeze(1).repeat(1, n_paths, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

            x_token = tokenizer.encode(x, half=True)

            initial_seq_len = x.size(1)
            batch_size = x_token[0].size(0)
            total_seq_len = initial_seq_len + pred_len
            full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

            generated_pre = x_token[0].new_empty(batch_size, pred_len)
            generated_post = x_token[1].new_empty(batch_size, pred_len)

            pre_buffer = x_token[0].new_zeros(batch_size, max_context)
            post_buffer = x_token[1].new_zeros(batch_size, max_context)
            buffer_len = min(initial_seq_len, max_context)
            if buffer_len > 0:
                start_idx = max(0, initial_seq_len - max_context)
                pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
                post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

            for i in range(pred_len):
                if verbose and (i % 10 == 0 or i == pred_len - 1):
                    print(f"    step {i + 1}/{pred_len}")
                current_seq_len = initial_seq_len + i
                window_len = min(current_seq_len, max_context)

                if current_seq_len <= max_context:
                    input_tokens = [
                        pre_buffer[:, :window_len],
                        post_buffer[:, :window_len]
                    ]
                else:
                    input_tokens = [pre_buffer, post_buffer]

                context_end = current_seq_len
                context_start = max(0, context_end - max_context)
                current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

                s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
                s1_logits = s1_logits[:, -1, :]
                sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

                s2_logits = model.decode_s2(context, sample_pre)
                s2_logits = s2_logits[:, -1, :]
                sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

                generated_pre[:, i] = sample_pre.squeeze(-1)
                generated_post[:, i] = sample_post.squeeze(-1)

                if current_seq_len < max_context:
                    pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                    post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
                else:
                    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                    pre_buffer[:, -1] = sample_pre.squeeze(-1)
                    post_buffer[:, -1] = sample_post.squeeze(-1)

            full_pre = torch.cat([x_token[0], generated_pre], dim=1)
            full_post = torch.cat([x_token[1], generated_post], dim=1)

            context_start = max(0, total_seq_len - max_context)
            input_tokens = [
                full_pre[:, context_start:total_seq_len].contiguous(),
                full_post[:, context_start:total_seq_len].contiguous()
            ]
            z = tokenizer.decode(input_tokens, half=True)
            # z is (n_paths, max_context, n_features): the input batch of 1 was
            # repeated to n_paths, so the batch dim IS the path index. (The
            # original auto_regressive_inference reshapes to (1, n, ...) only
            # because it averages the paths; we keep them separate.)
            return z.cpu().numpy()

    finally:
        model.train(was_training)


def sample_paths(predictor: KronosPredictor, x_df: pd.DataFrame, x_timestamp: pd.Series,
                 y_timestamp: pd.Series, pred_len: int, n_paths: int,
                 T: float = T, top_k: int = 0, top_p: float = TOP_P, verbose: bool = False):
    """Draw `n_paths` independent stochastic forecast paths in one batched loop.

    Mirrors the preprocessing of KronosPredictor.predict (volume/amount fill,
    time-stamp features, z-score normalization, clipping) but returns the
    individual paths instead of their average, so we can build an empirical
    distribution (quantile bands, probability metrics).

    Returns:
        np.ndarray of shape (n_paths, pred_len, 6) in the original price scale,
        with channels [open, high, low, close, volume, amount].
    """
    price_cols = predictor.price_cols
    df = x_df.copy()
    if predictor.vol_col not in df.columns:
        df[predictor.vol_col] = 0.0
        df[predictor.amt_vol] = 0.0
    if predictor.amt_vol not in df.columns and predictor.vol_col in df.columns:
        df[predictor.amt_vol] = df[predictor.vol_col] * df[price_cols].mean(axis=1)

    x_time_df = calc_time_stamps(x_timestamp)
    y_time_df = calc_time_stamps(y_timestamp)

    x = df[price_cols + [predictor.vol_col, predictor.amt_vol]].values.astype(np.float32)
    x_stamp = x_time_df.values.astype(np.float32)
    y_stamp = y_time_df.values.astype(np.float32)

    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    x = np.clip((x - x_mean) / (x_std + 1e-5), -predictor.clip, predictor.clip)

    x_t = torch.from_numpy(x[np.newaxis, :]).to(predictor.device)
    x_stamp_t = torch.from_numpy(x_stamp[np.newaxis, :]).to(predictor.device)
    y_stamp_t = torch.from_numpy(y_stamp[np.newaxis, :]).to(predictor.device)

    preds = _auto_regressive_paths(
        predictor.tokenizer, predictor.model, x_t, x_stamp_t, y_stamp_t,
        predictor.max_context, pred_len, predictor.clip, T, top_k, top_p,
        n_paths, verbose=verbose,
    )
    # preds: (n_paths, max_context, n_features) — the final decode returns the
    # last max_context tokens of the full sequence, so the forecast horizon is
    # its last pred_len elements (same convention as KronosPredictor.generate).
    preds = preds[:, -pred_len:, :]
    # Denormalize back to the original data scale.
    preds = preds * (x_std + 1e-5) + x_mean
    return preds


# ---------------------------------------------------------------------------
# Volatility calibration (post-hoc vol-targeting)
# ---------------------------------------------------------------------------
def calibrate_vol(paths: np.ndarray, hist_close: np.ndarray,
                  vol_lookback: int = VOL_LOOKBACK, max_scale: float = 5.0,
                  shape: bool = True) -> np.ndarray:
    """Rescale each path's deviation from the ensemble mean so the ensemble's
    per-horizon spread matches the asset's recent realized volatility.

    Kronos is trained to reconstruct price levels, so its sampled paths are
    too smooth: the ensemble's day-to-day dispersion comes out ~2-5x smaller
    than gold's realized vol, which makes the quantile bands, VaR/CVaR and
    up/down probabilities unrealistically tight. This post-hoc fix rescales
    each path's DEVIATION FROM THE ENSEMBLE MEAN (never the mean itself), so:

      * the ensemble mean path is unchanged -> MAPE / bias are unaffected;
      * the per-horizon spread is driven to sigma_d * sqrt(t) (Brownian
        scaling), where sigma_d is the std of the last `vol_lookback`
        realized daily returns (recent regime, not the full lookback);
      * all four price channels are scaled by the same per-horizon factor
        (each around its own channel mean), so the OHLC envelope is kept.

    With `shape=True` (default) the fix also corrects the ensemble's SHAPE,
    not just its scale. A near-degenerate ensemble is concentrated near the
    mean, so a single linear scale cannot fix both the inner (25-75%) and
    outer (5-95%) bands at once. Instead each path's close is ranked and
    mapped to its standard-normal quantile z = Phi^-1((rank - 0.5)/n)
    (standardized to mean 0 / std 1), then every price channel is rebuilt
    as channel_mean + z * channel_std * scale. The shared close-rank shock
    keeps OHLC coherent, the ensemble is exactly normal-shaped at the
    calibrated std (so the 5-95% / 25-75% bands contain 90% / 50% of mass
    by construction), and the mean path is preserved exactly (mean of z = 0).
    With `shape=False` the original pure linear scale is applied.

    The scale factor is capped at `max_scale` to avoid amplifying sampling
    noise when the model emits near-identical paths.
    """
    hist = np.asarray(hist_close, dtype=float)
    h = hist[-(vol_lookback + 1):]
    if len(h) < 3:
        print("⚠️  Vol calibration skipped: not enough history.")
        return paths
    r = np.diff(h) / h[:-1]
    sigma_d = float(np.std(r))
    if sigma_d <= 0:
        print("⚠️  Vol calibration skipped: zero realized vol in history.")
        return paths

    n, H, _ = paths.shape
    mean_close = paths[:, :, CLOSE_IDX].mean(axis=0)
    t = np.arange(1, H + 1, dtype=float)

    # Standardized normal-quantile table for an ensemble of size n: path of
    # rank k (1-based) maps to Phi^-1((k - 0.5)/n), standardized to mean 0 /
    # std 1. Fixed for a given n, so compute the n inv_cdf calls once.
    k = np.arange(1, n + 1)
    z_table = np.array([NORMAL.inv_cdf((ki - 0.5) / n) for ki in k])
    z_table = (z_table - z_table.mean()) / z_table.std()

    out = paths.copy()
    scales = []
    for i in range(H):
        cur = float(np.std(paths[:, i, CLOSE_IDX] - mean_close[i]) / mean_close[i])
        target = sigma_d * np.sqrt(t[i])
        s = min(target / cur, max_scale) if cur > 1e-8 else 0.0
        scales.append(s)
        if shape:
            # Probability integral transform: rank the ensemble's closes and
            # map each path to its normal quantile (stable sort keeps ties
            # in their original order).
            order = np.argsort(paths[:, i, CLOSE_IDX], kind="stable")
            rank_pos = np.empty(n, dtype=int)
            rank_pos[order] = np.arange(n)
            z = z_table[rank_pos]
            for c in PRICE_IDX:
                ch = out[:, i, c]
                m = float(ch.mean())
                sc = float(ch.std())
                out[:, i, c] = m + z * sc * s
        else:
            for c in PRICE_IDX:
                ch = out[:, i, c]
                m = float(ch.mean())
                out[:, i, c] = m + (ch - m) * s

    cur_final = float(np.std(paths[:, -1, CLOSE_IDX] - mean_close[-1]) / mean_close[-1])
    shape_note = " + normal-quantile shape" if shape else ""
    print(f"🎯 Vol calibration: sigma_d (last {vol_lookback}d) = {sigma_d * 100:.3f}%/day; "
          f"final-horizon spread {cur_final * 100:.2f}% -> {cur_final * scales[-1] * 100:.2f}% "
          f"(target {sigma_d * np.sqrt(H) * 100:.2f}%, scale {scales[-1]:.2f}x{shape_note})")
    return out


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------
def summarize_paths(paths: np.ndarray, y_timestamp: pd.Series) -> pd.DataFrame:
    """Per-step distribution summary of the ensemble close prices."""
    closes = paths[:, :, CLOSE_IDX]  # (n_paths, pred_len)
    return pd.DataFrame({
        "mean": closes.mean(axis=0),
        "std": closes.std(axis=0),
        "median": np.median(closes, axis=0),
        "lower": np.quantile(closes, 0.05, axis=0),
        "upper": np.quantile(closes, 0.95, axis=0),
        "mid_low": np.quantile(closes, 0.25, axis=0),
        "mid_high": np.quantile(closes, 0.75, axis=0),
    }, index=pd.DatetimeIndex(y_timestamp.values, name="timestamps"))


def probability_metrics(paths: np.ndarray, ref_price: float, var_level: float = 0.05) -> dict:
    """Key probability metrics for the final forecast step.

    ref_price: the last observed close (the reference for up/down moves).
    """
    closes = paths[:, :, CLOSE_IDX]
    final = closes[:, -1]

    final_mean = float(final.mean())
    final_std = float(final.std())
    upside_prob = float((final > ref_price).mean())
    downside_prob = float((final < ref_price).mean())

    var_value = float(np.quantile(final, var_level))
    worst = final[final <= var_value]
    cvar_value = float(worst.mean()) if len(worst) > 0 else var_value

    # Worst intra-path drawdown across the ensemble.
    max_dd = 0.0
    for p in range(closes.shape[0]):
        prices = closes[p]
        running_max = np.maximum.accumulate(prices)
        drawdown = (prices - running_max) / running_max
        max_dd = min(max_dd, float(drawdown.min()))

    return {
        "final_mean": final_mean,
        "final_std": final_std,
        "upside_prob": upside_prob,
        "downside_prob": downside_prob,
        "VaR": var_value,
        "CVaR": cvar_value,
        "max_drawdown": max_dd,
    }


# ---------------------------------------------------------------------------
# Rolling walk-forward backtest
# ---------------------------------------------------------------------------
def run_backtest(predictor: KronosPredictor, df: pd.DataFrame, lookback: int,
                 pred_len: int, n_windows: int, n_paths: int,
                 T: float, top_k: int, top_p: float,
                 vol_calibration: bool = True, vol_shape: bool = True,
                 up_thresh: float = 0.02, down_thresh: float = -0.02):
    """Rolling walk-forward backtest over the tail of the file.

    Window w uses history ending `w * pred_len` rows before the file end and
    is evaluated against the file's real dates for the following `pred_len`
    rows (in-sample mode). The tail of the 10y file lies inside the
    fine-tuning validation split, so this is the most honest evaluation the
    file allows.

    Per window we track: MAPE, signed bias, Pearson correlation of daily
    close-to-close returns, direction accuracy, realized-vol ratio, prediction-
    interval coverage (fraction of actual closes inside the 5-95% / 25-75%
    bands — a well-calibrated 5-95% band should contain ~90%), and the window
    regime (up / down / sideways by the window's actual return).

    Returns the metrics DataFrame and saves to outputs/: the per-window
    metrics CSV, the per-day predicted-vs-actual CSV, a pooled correlation
    report (see _correlation_report) + scatter, and the per-window chart.
    """
    rows = []
    daily_rows = []       # per-day predicted vs actual, for the correlation report
    window_segments = []  # (y_timestamp, pred_close, actual_close, summary) for the chart

    for w in range(n_windows):
        end_offset = w * pred_len
        x_df, x_timestamp, y_timestamp = prepare_inputs(
            df, lookback, pred_len, forward=False, end_offset=end_offset
        )
        print(f"🔁 Backtest window {w + 1}/{n_windows}: "
              f"{y_timestamp.iloc[0].date()} ~ {y_timestamp.iloc[-1].date()} ...")

        paths = sample_paths(predictor, x_df, x_timestamp, y_timestamp,
                             pred_len, n_paths, T=T, top_k=top_k, top_p=top_p)
        if vol_calibration:
            # Honest walk-forward: sigma_d is estimated from the history
            # available at this window's start (x_df), not the file tail.
            paths = calibrate_vol(paths, x_df["close"].values, shape=vol_shape)
        pred_close = paths[:, :, CLOSE_IDX].mean(axis=0)
        actual_close = df["close"].iloc[len(df) - pred_len - end_offset: len(df) - end_offset].values

        pred_ret = np.diff(pred_close) / pred_close[:-1]
        act_ret = np.diff(actual_close) / actual_close[:-1]

        mape = float(np.mean(np.abs(pred_close - actual_close) / actual_close) * 100)
        bias = float(np.mean((pred_close - actual_close) / actual_close) * 100)
        corr = float(np.corrcoef(pred_ret, act_ret)[0, 1]) if np.std(act_ret) > 0 else float("nan")
        dir_acc = float(np.mean(np.sign(pred_ret) == np.sign(act_ret)))
        vol_ratio = float(np.std(pred_ret) / np.std(act_ret)) if np.std(act_ret) > 0 else float("nan")
        pred_window_ret = pred_close[-1] / pred_close[0] - 1

        # Per-day records (full 30-day closes; returns are close-to-close within
        # the window, so the first day has none). Used by the correlation report.
        daily_rows.append(pd.DataFrame({
            "date": y_timestamp.values,
            "window": w + 1,
            "pred_close": pred_close,
            "actual_close": actual_close,
            "pred_ret": np.concatenate([[np.nan], pred_ret]),
            "act_ret": np.concatenate([[np.nan], act_ret]),
        }))

        window_ret = actual_close[-1] / actual_close[0] - 1
        if window_ret > up_thresh:
            regime = "up"
        elif window_ret < down_thresh:
            regime = "down"
        else:
            regime = "sideways"

        summary = summarize_paths(paths, y_timestamp)
        lo90, hi90 = summary["lower"].values, summary["upper"].values
        lo50, hi50 = summary["mid_low"].values, summary["mid_high"].values
        coverage_90 = float(np.mean((actual_close >= lo90) & (actual_close <= hi90)))
        coverage_50 = float(np.mean((actual_close >= lo50) & (actual_close <= hi50)))

        rows.append({
            "window": w + 1,
            "start": y_timestamp.iloc[0],
            "end": y_timestamp.iloc[-1],
            "regime": regime,
            "window_ret_pct": round(window_ret * 100, 2),
            "pred_window_ret_pct": round(pred_window_ret * 100, 2),
            "mape_pct": round(mape, 2),
            "bias_pct": round(bias, 2),
            "ret_corr": round(corr, 3),
            "dir_acc": round(dir_acc, 3),
            "vol_ratio": round(vol_ratio, 3),
            "coverage_90": round(coverage_90, 3),
            "coverage_50": round(coverage_50, 3),
        })
        window_segments.append((y_timestamp, pred_close, actual_close, summary))

    bt_df = pd.DataFrame(rows)
    out_csv = os.path.join(save_dir, "backtest_XAUUSD_1d.csv")
    bt_df.to_csv(out_csv, index=False)
    print(f"\n✅ Backtest metrics saved: {out_csv}")
    print(bt_df.to_string(index=False))

    daily_df = pd.concat(daily_rows, ignore_index=True)
    daily_csv = os.path.join(save_dir, "backtest_XAUUSD_1d_daily.csv")
    daily_df.to_csv(daily_csv, index=False)
    print(f"✅ Backtest per-day data saved: {daily_csv}")

    _correlation_report(daily_df, bt_df)

    _plot_backtest(df, window_segments, os.path.join(save_dir, "backtest_XAUUSD_1d_chart.png"))
    return bt_df


def _plot_backtest(df: pd.DataFrame, window_segments, out_path: str):
    """One chart: recent history + per-window actual vs predicted close with bands."""
    plt.figure(figsize=(14, 6))
    tail = df.iloc[-200:]
    plt.plot(tail["timestamps"], tail["close"], color="blue", label="Historical", linewidth=1.2)

    for i, (y_timestamp, pred_close, actual_close, summary) in enumerate(window_segments):
        idx = pd.DatetimeIndex(y_timestamp.values)
        plt.plot(idx, actual_close, color="blue", linewidth=1.2)
        plt.plot(idx, pred_close, color="red", linestyle="--", linewidth=1.2,
                 label="Predicted (mean)" if i == 0 else None)
        plt.fill_between(idx, summary["lower"], summary["upper"],
                         color="red", alpha=0.12,
                         label="5-95% band" if i == 0 else None)

    plt.title("Kronos XAUUSD (1D) rolling walk-forward backtest")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Backtest chart saved: {out_path}")


def _correlation_report(daily_df: pd.DataFrame, bt_df: pd.DataFrame):
    """Quantify the correlation between predicted and actual values.

    Pools the per-day records across all backtest windows and reports:
      * Pearson + Spearman correlation of daily close-to-close returns
        (the standard "information coefficient" in quant research);
      * within-window Pearson correlation of price levels (averaged over
        windows, to avoid a between-window level artifact from pooling);
      * correlation of daily |returns| (a volatility-timing signal);
      * correlation of 30-day cumulative (window) returns.
    Also saves a scatter of predicted vs actual daily returns.
    """
    pr = daily_df["pred_ret"].dropna()
    ar = daily_df["act_ret"].dropna()
    n = len(pr)
    pearson_ret = float(pr.corr(ar))
    # Spearman = Pearson on ranks (avoids a scipy dependency; pandas'
    # method="spearman" requires scipy, which is not installed here).
    spearman_ret = float(pr.rank().corr(ar.rank()))
    vol_corr = float(pr.abs().corr(ar.abs()))

    # Within-window level correlation (avoids pooling across different price
    # regimes, which would be dominated by between-window level differences).
    level_corr = float(np.mean(
        [float(g["pred_close"].corr(g["actual_close"])) for _, g in daily_df.groupby("window")]))

    # Window-level: 30-day cumulative return, predicted vs actual.
    win_corr = float(bt_df["pred_window_ret_pct"].corr(bt_df["window_ret_pct"]))

    print("\n📐 Correlation between predictions and actuals "
          f"(pooled over {n} daily observations, {daily_df['window'].nunique()} windows):")
    print(f"   daily return corr    (Pearson):    {pearson_ret:+.3f}")
    print(f"   daily return corr    (Spearman):   {spearman_ret:+.3f}")
    print(f"   price level corr     (within-win): {level_corr:+.3f}")
    print(f"   daily |return| corr  (vol timing): {vol_corr:+.3f}")
    print(f"   30-day window return corr:         {win_corr:+.3f}  "
          f"(n={len(bt_df)} windows)")

    # Scatter: predicted vs actual daily return, colored by window.
    d = daily_df.dropna(subset=["pred_ret", "act_ret"])
    plt.figure(figsize=(7, 6))
    for w, g in d.groupby("window"):
        plt.scatter(g["act_ret"] * 100, g["pred_ret"] * 100,
                    label=f"W{w}", alpha=0.7, s=28)
    lo = min(d[["act_ret", "pred_ret"]].min().min() * 100, 0)
    hi = max(d[["act_ret", "pred_ret"]].max().max() * 100, 0)
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    plt.xlabel("Actual daily return (%)")
    plt.ylabel("Predicted daily return (%)")
    plt.title(f"Predicted vs actual daily return\n"
              f"Pearson r = {pearson_ret:+.3f}   Spearman ρ = {spearman_ret:+.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(save_dir, "backtest_XAUUSD_1d_corr.png")
    plt.savefig(out)
    plt.close()
    print(f"📊 Correlation scatter saved: {out}")


# ---------------------------------------------------------------------------
# Main forecast
# ---------------------------------------------------------------------------
def plot_result(df_hist: pd.DataFrame, pred_df: pd.DataFrame, summary: pd.DataFrame,
                title: str, out_path: str, hist_tail: int = 250):
    plt.figure(figsize=(12, 6))
    # Zoom to recent history so the forecast is visible on the same scale
    # (plotting the full 10y range lets old price levels dominate the y-axis).
    hist = df_hist.iloc[-hist_tail:]
    plt.plot(hist["timestamps"], hist["close"], label="Historical", color="blue")
    plt.plot(pred_df.index, pred_df["close"], label="Predicted (mean of paths)",
             color="red", linestyle="--")
    plt.fill_between(summary.index, summary["lower"], summary["upper"],
                     color="red", alpha=0.15, label="5-95% band")
    plt.fill_between(summary.index, summary["mid_low"], summary["mid_high"],
                     color="red", alpha=0.30, label="25-75% band")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Chart saved: {out_path}")


def predict_future(csv_path: str, lookback: int, pred_len: int, forward: bool,
                   n_paths: int, T: float, top_k: int, top_p: float,
                   use_pretrained: bool, do_backtest: bool, n_backtest_windows: int,
                   vol_calibration: bool = True, vol_shape: bool = True):
    tok_name, model_name = resolve_model_paths(use_pretrained)
    print(f"🚀 Loading Kronos tokenizer: {tok_name}")
    print(f"   Loading Kronos model:      {model_name}")
    tokenizer = KronosTokenizer.from_pretrained(tok_name)
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)

    df = load_data(csv_path)
    x_df, x_timestamp, y_timestamp = prepare_inputs(df, lookback, pred_len, forward=forward)

    mode = "FORWARD (out-of-sample)" if forward else "IN-SAMPLE (evaluating the file's tail)"
    print(f"🔮 Sampling {n_paths} paths for a {pred_len}-day forecast "
          f"[{mode}, lookback={lookback}, T={T}, top_p={top_p}, device={DEVICE}] ...")
    paths = sample_paths(predictor, x_df, x_timestamp, y_timestamp,
                         pred_len, n_paths, T=T, top_k=top_k, top_p=top_p, verbose=True)

    if vol_calibration:
        paths = calibrate_vol(paths, x_df["close"].values, shape=vol_shape)

    mean_path = paths.mean(axis=0)  # (pred_len, 6)
    pred_df = pd.DataFrame(
        mean_path,
        columns=PRICE_COLS + ["volume", "amount"],
        index=pd.DatetimeIndex(y_timestamp.values, name="timestamps"),
    )
    summary = summarize_paths(paths, y_timestamp)

    print("\nForecasted Data Head (ensemble mean):")
    print(pred_df[PRICE_COLS].head().to_string())

    # Probability metrics for the final step, referenced to the last observed close.
    ref_price = float(df["close"].iloc[-1])
    metrics = probability_metrics(paths, ref_price)
    print("\n📈 Probability metrics (final forecast step):")
    print(f"   last observed close:      {ref_price:,.2f}")
    print(f"   final mean / std:         {metrics['final_mean']:,.2f} / {metrics['final_std']:,.2f}")
    print(f"   upside / downside prob:   {metrics['upside_prob']:.1%} / {metrics['downside_prob']:.1%}")
    print(f"   VaR(5%) / CVaR(5%):       {metrics['VaR']:,.2f} / {metrics['CVaR']:,.2f}")
    print(f"   worst path drawdown:      {metrics['max_drawdown']:.2%}")

    # Save history + mean forecast.
    df_out = pd.concat(
        [
            df[["timestamps"] + PRICE_COLS].rename(columns={"timestamps": "date"}),
            pred_df.reset_index().rename(columns={"timestamps": "date"}),
        ],
        ignore_index=True,
    )
    out_file = os.path.join(save_dir, "pred_XAUUSD_1d_data.csv")
    df_out.to_csv(out_file, index=False)
    print(f"✅ Prediction saved: {out_file}")

    plot_result(df, pred_df, summary,
                f"Kronos Prediction for XAUUSD (1D) — {mode}",
                os.path.join(save_dir, "pred_XAUUSD_1d_chart.png"))

    if do_backtest:
        print(f"\n🔁 Running rolling walk-forward backtest ({n_backtest_windows} windows) ...")
        run_backtest(predictor, df, lookback, pred_len, n_backtest_windows,
                     n_paths, T=T, top_k=top_k, top_p=top_p,
                     vol_calibration=vol_calibration, vol_shape=vol_shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos XAUUSD daily prediction (Monte Carlo + backtest)")
    default_csv = os.path.join(REPO_ROOT, "data", "XAUUSD_1d_10y.csv")
    parser.add_argument("--csv", default=default_csv, help="Input CSV path")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="History window (<= max_context)")
    parser.add_argument("--pred-len", type=int, default=PRED_LEN, help="Number of days to forecast")
    parser.add_argument("--forward", action=argparse.BooleanOptionalAction, default=True,
                        help="Forecast forward from the last date (default). "
                             "Use --no-forward for in-sample evaluation on the file's tail.")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use the HF pre-trained Kronos models instead of the fine-tuned checkpoints")
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT,
                        help="Monte Carlo ensemble size")
    parser.add_argument("--temperature", type=float, default=T, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=TOP_P, help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k filtering (0 = off)")
    parser.add_argument("--vol-calibration", action=argparse.BooleanOptionalAction, default=True,
                        help="Rescale each path's deviation from the ensemble mean so the "
                             "per-horizon spread matches recent realized vol (default on). "
                             "Use --no-vol-calibration to disable.")
    parser.add_argument("--vol-shape", action=argparse.BooleanOptionalAction, default=True,
                        help="Also map the ensemble to a normal shape (rank -> Phi^-1 quantile) "
                             "so the 25-75%% / 5-95%% bands contain 50%% / 90%% of mass (default on). "
                             "Use --no-vol-shape for the original pure linear scale.")
    parser.add_argument("--no-backtest", action="store_true",
                        help="Skip the rolling walk-forward backtest")
    parser.add_argument("--backtest-windows", type=int, default=5,
                        help="Number of rolling backtest windows (step = pred-len)")
    args = parser.parse_args()

    predict_future(
        csv_path=args.csv,
        lookback=args.lookback,
        pred_len=args.pred_len,
        forward=args.forward,
        n_paths=args.sample_count,
        T=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_pretrained=args.pretrained,
        do_backtest=not args.no_backtest,
        n_backtest_windows=args.backtest_windows,
        vol_calibration=args.vol_calibration,
        vol_shape=args.vol_shape,
    )
