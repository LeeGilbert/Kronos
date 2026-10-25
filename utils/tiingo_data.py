"""
Tiingo k-line (candlestick) data downloader for forex symbols (e.g., XAUUSD).

API docs: https://www.tiingo.com/documentation/general/connecting
Endpoint: GET https://api.tiingo.com/tiingo/daily/{ticker}/prices
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

TIINGO_BASE_URL = "https://api.tiingo.com"

# Map user-friendly frequency names to Tiingo endpoint segments
FREQUENCY_MAP = {
    "1d": "daily",
    "1w": "weekly",
    "1mo": "monthly",
    "1h": "hourly",
    "4h": "4hour",
    "1m": "minute",
    "5m": "5minute",
    "15m": "15minute",
    "30m": "30minute",
}

# Columns to keep in the output DataFrame
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


@dataclass
class KlineConfig:
    """Configuration for a k-line download."""

    symbol: str = "XAUUSD"
    frequency: str = "1d"
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    def __post_init__(self):
        if self.frequency not in FREQUENCY_MAP:
            raise ValueError(
                f"Invalid frequency '{self.frequency}'. "
                f"Must be one of: {list(FREQUENCY_MAP.keys())}"
            )


class TiingoClient:
    """Client for downloading k-line data from Tiingo."""

    def __init__(self, api_token: Optional[str] = None, base_url: str = TIINGO_BASE_URL):
        self.api_token = api_token or os.environ.get("TIINGO_API_TOKEN", "")
        if not self.api_token:
            raise ValueError(
                "No API token provided. Pass api_token or set TIINGO_API_TOKEN env var."
            )
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {self.api_token}",
                "Content-Type": "application/json",
                "User-Agent": "kronos/1.0",
            }
        )

    def _get_prices(
        self,
        symbol: str,
        frequency: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """Fetch k-line data from the Tiingo API for a date range."""
        endpoint = FREQUENCY_MAP[frequency]
        url = f"{self.base_url}/tiingo/{endpoint}/{symbol}/prices"

        params = {
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": end.strftime("%Y-%m-%d"),
            "format": "json",
        }

        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def download_klines(
        self,
        config: Optional[KlineConfig] = None,
        *,
        symbol: str = "XAUUSD",
        frequency: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        sleep_between: float = 1.0,
    ) -> pd.DataFrame:
        """
        Download k-line data, automatically paginating through the full range.

        Parameters
        ----------
        config : KlineConfig, optional
            Pre-built config (overrides individual params if provided).
        symbol : str
            Ticker symbol, e.g. "XAUUSD".
        frequency : str
            Candle frequency: 1d, 1w, 1mo, 1h, 4h, 1m, 5m, 15m, 30m.
        start : datetime, optional
            Start of the range (default: 1 year ago).
        end : datetime, optional
            End of the range (default: now).
        sleep_between : float
            Seconds to sleep between paginated requests.

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume
            Indexed by date (UTC).
        """
        if config is not None:
            symbol = config.symbol
            frequency = config.frequency
            start = config.start
            end = config.end

        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - pd.DateOffset(years=1)

        # Ensure timezone-aware
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        logger.info(f"Downloading {symbol} {frequency} klines: {start.date()} → {end.date()}")

        all_rows: list[dict] = []
        current_start = start

        # Paginate in 1-year chunks to avoid hitting API limits
        while current_start < end:
            chunk_end = min(current_start + pd.DateOffset(years=1), end)

            rows = self._get_prices(
                symbol=symbol,
                frequency=frequency,
                start=current_start,
                end=chunk_end,
            )

            if not rows:
                logger.debug(f"  No data for {current_start.date()} → {chunk_end.date()}")
                current_start = chunk_end + pd.Timedelta(days=1)
                continue

            all_rows.extend(rows)
            logger.debug(f"  Fetched {len(rows)} candles (total: {len(all_rows)})")

            current_start = chunk_end + pd.Timedelta(days=1)
            time.sleep(sleep_between)

        if not all_rows:
            logger.warning(f"No data returned for {symbol} {frequency}")
            return pd.DataFrame(columns=OHLCV_COLUMNS[1:]).set_index("date")

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)

        # Keep only OHLCV columns
        df = df[OHLCV_COLUMNS]
        df = df.set_index("date").sort_index()

        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]

        logger.info(f"Downloaded {len(df)} candles for {symbol} {frequency}")
        return df

    def download_xauusd(
        self,
        frequency: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Convenience method to download XAUUSD k-lines."""
        return self.download_klines(
            symbol="XAUUSD",
            frequency=frequency,
            start=start,
            end=end,
            **kwargs,
        )


def download_xauusd(
    api_token: Optional[str] = None,
    frequency: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    One-shot helper to download XAUUSD k-line data from Tiingo.

    Parameters
    ----------
    api_token : str, optional
        Tiingo API token. Falls back to TIINGO_API_TOKEN env var.
    frequency : str
        Candle frequency (default: "1d").
    start : datetime, optional
        Start date (default: 1 year ago).
    end : datetime, optional
        End date (default: now).

    Returns
    -------
    pd.DataFrame
        OHLCV data indexed by UTC date.

    Example
    -------
    >>> df = download_xauusd(
    ...     api_token="your_token",
    ...     frequency="1d",
    ...     start=datetime(2024, 1, 1),
    ... )
    >>> print(df.tail())
    """
    client = TiingoClient(api_token)
    return client.download_xauusd(
        frequency=frequency, start=start, end=end, **kwargs
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download XAUUSD k-line data from Tiingo"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Tiingo API token (or set TIINGO_API_TOKEN env var)",
    )
    parser.add_argument(
        "--symbol",
        default="XAUUSD",
        help="Ticker symbol (default: XAUUSD)",
    )
    parser.add_argument(
        "--frequency",
        default="1d",
        choices=list(FREQUENCY_MAP.keys()),
        help="Candle frequency (default: 1d)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date (ISO format, e.g. 2024-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (ISO format, e.g. 2024-12-31)",
    )
    parser.add_argument(
        "--output",
        default="xauusd_klines.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start) if args.start else None
    end_dt = datetime.fromisoformat(args.end) if args.end else None

    client = TiingoClient(args.token)
    df = client.download_klines(
        symbol=args.symbol,
        frequency=args.frequency,
        start=start_dt,
        end=end_dt,
    )

    df.to_csv(args.output)
    print(f"Saved {len(df)} candles to {args.output}")
    print(df.tail())
