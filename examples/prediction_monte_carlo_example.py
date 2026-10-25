"""
Monte Carlo forecasting with Kronos.

Instead of a single point forecast, this example draws N independent stochastic
paths from the Kronos model (via the low-level `generate` API, which returns
every sampled path rather than averaging them) and treats the ensemble as an
empirical distribution of possible futures.

From that distribution we derive:
  * the mean forecast (solid line),
  * an uncertainty band (shaded area, e.g. 5-95% / 25-75% quantiles),
  * key probability metrics (volatility, upside/downside probability,
    VaR / CVaR, and the probability of a price target being hit).

Run:
    python examples/prediction_monte_carlo_example.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
from model.kronos import calc_time_stamps


# ---------------------------------------------------------------------------
# Monte Carlo sampling
# ---------------------------------------------------------------------------
def monte_carlo_paths(predictor, df, x_timestamp, y_timestamp, pred_len,
                      n_paths=30, T=1.0, top_k=0, top_p=0.9, verbose=False):
    """
    Draw `n_paths` independent stochastic forecast paths from the model.

    This mirrors the preprocessing that `KronosPredictor.predict` performs
    (normalization + time-stamp features) but calls `predictor.generate`
    directly, once per path, so that the individual sampled paths are preserved
    instead of being averaged into a single point forecast.

    Args:
        predictor: A configured KronosPredictor instance.
        df: Historical OHLCV(+amount) DataFrame (the lookback window).
        x_timestamp: Datetime index/series aligned to `df`.
        y_timestamp: Datetime index/series of length `pred_len` for the future.
        pred_len: Number of future steps to forecast.
        n_paths: Number of Monte Carlo paths to sample.
        T, top_k, top_p: Stochastic sampling controls.

    Returns:
        pd.DataFrame: One row per (path, step). Columns are the model outputs
        (open, high, low, close, volume, amount) plus a `path` column. The
        index is a MultiIndex of (path, timestamp).
    """
    price_cols = predictor.price_cols
    vol_col = predictor.vol_col
    amt_col = predictor.amt_vol

    df = df.copy()
    if vol_col not in df.columns:
        df[vol_col] = 0.0
        df[amt_col] = 0.0
    if amt_col not in df.columns and vol_col in df.columns:
        df[amt_col] = df[vol_col] * df[price_cols].mean(axis=1)

    x_time_df = calc_time_stamps(x_timestamp)
    y_time_df = calc_time_stamps(y_timestamp)

    x = df[price_cols + [vol_col, amt_col]].values.astype(np.float32)
    x_stamp = x_time_df.values.astype(np.float32)
    y_stamp = y_time_df.values.astype(np.float32)

    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    x = (x - x_mean) / (x_std + 1e-5)
    x = np.clip(x, -predictor.clip, predictor.clip)

    # Batch dimension of 1.
    x = x[np.newaxis, :]
    x_stamp = x_stamp[np.newaxis, :]
    y_stamp = y_stamp[np.newaxis, :]

    # `predictor.generate` internally AVERAGES over `sample_count`, so it can
    # only ever return a single (averaged) path. To obtain N *independent*
    # stochastic paths we therefore sample one path at a time (sample_count=1)
    # and repeat, then stack the results into an ensemble.
    paths = []
    for _ in range(n_paths):
        preds = predictor.generate(x, x_stamp, y_stamp, pred_len,
                                   T=T, top_k=top_k, top_p=top_p,
                                   sample_count=1, verbose=verbose)
        # preds: (1, pred_len, n_features) -> squeeze the leading batch dim.
        paths.append(preds[0])

    preds = np.stack(paths, axis=0)  # (n_paths, pred_len, n_features)

    # Denormalize back to the original data scale.
    preds = preds * (x_std + 1e-5) + x_mean

    n_paths, pred_len_, n_features = preds.shape
    columns = price_cols + [vol_col, amt_col]

    records = []
    for p in range(n_paths):
        for t in range(pred_len_):
            row = {"path": p, "timestamp": y_timestamp.values[t]}
            for j, col in enumerate(columns):
                row[col] = preds[p, t, j]
            records.append(row)

    out = pd.DataFrame(records)
    out = out.set_index(["path", "timestamp"])
    return out


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------
def summarize_distribution(paths_df, close_col="close",
                           lower_q=0.05, upper_q=0.95,
                           mid_low_q=0.25, mid_high_q=0.75):
    """
    Aggregate the Monte Carlo paths into a per-step distribution summary.

    Returns a DataFrame indexed by timestamp with:
        mean, std, min, max,
        lower/upper (outer uncertainty band),
        mid_low/mid_high (inner uncertainty band),
        median.
    """
    grouped = paths_df[close_col].groupby(level="timestamp")
    summary = pd.DataFrame({
        "mean": grouped.mean(),
        "std": grouped.std(),
        "min": grouped.min(),
        "max": grouped.max(),
        "median": grouped.median(),
        "lower": grouped.quantile(lower_q),
        "upper": grouped.quantile(upper_q),
        "mid_low": grouped.quantile(mid_low_q),
        "mid_high": grouped.quantile(mid_high_q),
    })
    return summary


def probability_metrics(paths_df, close_col="close",
                         target_price=None, var_level=0.05):
    """
    Compute key probability metrics from the ensemble of paths.

    Metrics are reported for the final forecast step (the horizon of interest):
        * final_mean / final_std
        * upside_prob:  P(final price > starting price)
        * downside_prob: P(final price < starting price)
        * target_prob:  P(final price > target_price)  (if provided)
        * VaR:          the `var_level` quantile of the final price
        * CVaR:         mean of the worst `var_level` fraction of outcomes
        * max_drawdown: worst intra-path drawdown across the ensemble

    Returns a dict of scalar metrics.
    """
    # Per-path final price and per-path return relative to the last observed price.
    final_prices = paths_df[close_col].groupby(level="path").last()
    start_price = paths_df[close_col].groupby(level="path").first()

    final_mean = float(final_prices.mean())
    final_std = float(final_prices.std())

    # Use the mean starting price as the reference for up/down moves.
    ref_price = float(start_price.mean())
    upside_prob = float((final_prices > ref_price).mean())
    downside_prob = float((final_prices < ref_price).mean())

    target_prob = None
    if target_price is not None:
        target_prob = float((final_prices > target_price).mean())

    var_value = float(final_prices.quantile(var_level))
    worst = final_prices[final_prices <= var_value]
    cvar_value = float(worst.mean()) if len(worst) > 0 else var_value

    # Worst intra-path drawdown across all paths.
    max_dd = 0.0
    for path_id, grp in paths_df[close_col].groupby(level="path"):
        prices = grp.values
        running_max = np.maximum.accumulate(prices)
        drawdown = (prices - running_max) / running_max
        max_dd = min(max_dd, float(drawdown.min()))

    return {
        "final_mean": final_mean,
        "final_std": final_std,
        "upside_prob": upside_prob,
        "downside_prob": downside_prob,
        "target_prob": target_prob,
        "VaR": var_value,
        "CVaR": cvar_value,
        "max_drawdown": max_dd,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_distribution(kline_df, summary, paths_df, close_col="close"):
    """
    Plot the historical series together with the Monte Carlo forecast:
      * faint lines for a sample of individual paths,
      * a shaded uncertainty band (outer + inner quantiles),
      * the mean forecast as a solid line,
      * the ground-truth continuation (if available) for reference.
    """
    hist_close = kline_df[close_col]
    # Use the real timestamps as the x-axis (NOT the integer row index).
    # Plotting history against its integer index while plotting the forecast
    # against real datetimes makes matplotlib render the history near the
    # epoch (1970) and the forecast far to the right, creating a spurious gap.
    hist_index = kline_df["timestamps"]

    # Align the summary to the forecast timestamps.
    summary = summary.reindex(summary.index)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Historical ground truth.
    ax.plot(hist_index, hist_close.values, color="blue",
            linewidth=1.5, label="History")

    # Ground-truth continuation, if the data contains it.
    if len(kline_df) > len(hist_close) - summary.shape[0]:
        gt = kline_df[close_col].iloc[-summary.shape[0]:]
        if len(gt) == summary.shape[0]:
            ax.plot(summary.index, gt.values, color="green",
                    linewidth=1.5, linestyle="--", label="Ground Truth")

    # A handful of representative individual paths (faint).
    sample_paths = paths_df[close_col].groupby(level="path").last().sample(
        min(10, paths_df[close_col].groupby(level="path").ngroups),
        random_state=0)
    for path_id in sample_paths.index:
        path_series = paths_df.loc[path_id, close_col]
        ax.plot(summary.index, path_series.values, color="gray",
                alpha=0.25, linewidth=0.8)

    # Uncertainty bands.
    ax.fill_between(summary.index, summary["lower"], summary["upper"],
                    color="red", alpha=0.15, label="90% range (5-95%)")
    ax.fill_between(summary.index, summary["mid_low"], summary["mid_high"],
                    color="red", alpha=0.30, label="50% range (25-75%)")

    # Mean forecast.
    ax.plot(summary.index, summary["mean"], color="red",
            linewidth=2.0, label="Mean Forecast")

    ax.set_ylabel("Close Price", fontsize=14)
    ax.set_title("Kronos Monte Carlo Forecast", fontsize=16)
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load Model and Tokenizer
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

    # 2. Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # 3. Prepare Data
    df = pd.read_csv("./data/XSHG_5min_600977.csv")
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    lookback = 400
    pred_len = 60

    x_df = df.loc[:lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.loc[:lookback - 1, "timestamps"]
    y_timestamp = df.loc[lookback:lookback + pred_len - 1, "timestamps"]

    # 4. Monte Carlo sampling: N independent stochastic paths.
    n_paths = 30
    print(f"Sampling {n_paths} Monte Carlo paths ...")
    paths_df = monte_carlo_paths(
        predictor,
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        n_paths=n_paths,
        T=1.0,
        top_k=0,
        top_p=0.9,
        verbose=True,
    )

    # 5. Derived insights from the distribution.
    summary = summarize_distribution(paths_df, close_col="close")

    last_observed_close = float(x_df["close"].iloc[-1])
    metrics = probability_metrics(paths_df, close_col="close",
                                  target_price=last_observed_close * 1.02,
                                  var_level=0.05)

    print("\n=== Distribution Summary (per-step, close) ===")
    print(summary.describe().to_string())

    print("\n=== Key Probability Metrics (final step) ===")
    print(f"  Last observed close      : {last_observed_close:.4f}")
    print(f"  Final mean forecast      : {metrics['final_mean']:.4f}")
    print(f"  Final std (uncertainty)  : {metrics['final_std']:.4f}")
    print(f"  P(upside)                : {metrics['upside_prob']:.2%}")
    print(f"  P(downside)              : {metrics['downside_prob']:.2%}")
    if metrics["target_prob"] is not None:
        print(f"  P(price > +2% target)   : {metrics['target_prob']:.2%}")
    print(f"  VaR (5%)                 : {metrics['VaR']:.4f}")
    print(f"  CVaR (5%)                : {metrics['CVaR']:.4f}")
    print(f"  Worst intra-path drawdown: {metrics['max_drawdown']:.2%}")

    # 6. Visualize: history + ground truth + distribution.
    kline_df = df.loc[:lookback + pred_len - 1]
    plot_distribution(kline_df, summary, paths_df, close_col="close")


if __name__ == "__main__":
    main()
