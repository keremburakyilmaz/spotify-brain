import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


DEFAULT_LISTENING_TIMEZONE = "Europe/Istanbul"


def get_listening_timezone() -> ZoneInfo:
    timezone_name = os.getenv("LISTENING_TIMEZONE", DEFAULT_LISTENING_TIMEZONE)
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Invalid LISTENING_TIMEZONE '{timezone_name}'. Use an IANA timezone name."
        ) from exc


def to_listening_time(value):
    """Convert Spotify's UTC timestamps to the configured listening timezone."""
    timezone = get_listening_timezone()
    if isinstance(value, pd.Series):
        return pd.to_datetime(value, utc=True).dt.tz_convert(timezone)
    return pd.to_datetime(value, utc=True).tz_convert(timezone)


def listening_now() -> datetime:
    return datetime.now(get_listening_timezone())
