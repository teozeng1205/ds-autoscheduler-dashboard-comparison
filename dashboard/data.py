"""Data loading utilities for the hourly collection plans dashboard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .db import HourlyCollectionPlansReader, ActualSentDataReader

log = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).resolve().parents[1]
_CACHE_DIR = _ROOT_DIR / "resources" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path_for_ids(auto_schedule_ids: List[int]) -> Path:
    """Generate a cache filename based on the auto_schedule_ids."""
    ids_str = '_'.join(str(id) for id in sorted(auto_schedule_ids))
    return _CACHE_DIR / f"hcp_{ids_str}.csv"


def _load_cached_dataset(auto_schedule_ids: List[int]) -> Optional[pd.DataFrame]:
    """Load cached dataset if it exists."""
    cache_path = _cache_path_for_ids(auto_schedule_ids)
    if not cache_path.exists():
        return None
    try:
        df = pd.read_csv(cache_path)
        # Convert date column to datetime
        if 'plan_date' in df.columns:
            df['plan_date'] = pd.to_numeric(df['plan_date'], errors='coerce').astype('Int64')
        if 'plan_datetime' in df.columns:
            df['plan_datetime'] = pd.to_datetime(df['plan_datetime'], errors='coerce')
        return df
    except Exception as exc:
        log.warning("Unable to load cached dataset %s (%s).", cache_path, exc)
        return None


def _persist_dataset(df: pd.DataFrame, auto_schedule_ids: List[int]) -> None:
    """Persist dataset to cache."""
    cache_path = _cache_path_for_ids(auto_schedule_ids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(cache_path, index=False)
        log.info("Cached dataset to %s", cache_path)
    except Exception as exc:
        log.warning("Unable to persist dataset cache to %s: %s", cache_path, exc)


def get_auto_schedule_ids(reader: Optional[HourlyCollectionPlansReader] = None, limit: int = 10) -> List[int]:
    """Get the most recent auto_schedule_id values."""
    if reader is None:
        reader = HourlyCollectionPlansReader()
    return reader.fetch_auto_schedule_ids(limit=limit)


class DataLoader:
    """Loader responsible for preparing dashboard data."""

    def __init__(
        self,
        reader: HourlyCollectionPlansReader | None = None,
        auto_schedule_ids: Optional[List[int]] = None,
        num_ids: int = 2
    ):
        self.reader = reader or HourlyCollectionPlansReader()
        self.auto_schedule_ids = auto_schedule_ids
        self.num_ids = num_ids
        self._dataset: pd.DataFrame | None = None
        self._loaded_from_csv = False

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Return the prepared dataset, optionally forcing a refresh."""
        if force_refresh or self._dataset is None:
            self._dataset = self._build_dataset(force_refresh=force_refresh)
        return self._dataset.copy()

    @property
    def loaded_from_cache(self) -> bool:
        """Return True if the last load came from a cached CSV."""
        return self._loaded_from_csv

    def refresh(self) -> pd.DataFrame:
        """Force a rebuild of the underlying dataset and return it."""
        return self.load(force_refresh=True)

    def _build_dataset(self, force_refresh: bool = False) -> pd.DataFrame:
        log.info("Loading hourly collection plans dataset")

        # Get auto_schedule_ids if not provided
        if self.auto_schedule_ids is None:
            self.auto_schedule_ids = self.reader.fetch_auto_schedule_ids(limit=self.num_ids)

        if not self.auto_schedule_ids:
            raise RuntimeError("No auto_schedule_ids available")

        self._loaded_from_csv = False
        if not force_refresh:
            cached_df = _load_cached_dataset(self.auto_schedule_ids)
            if cached_df is not None:
                log.info("Loaded dataset from cache for auto_schedule_ids %s", self.auto_schedule_ids)
                self._loaded_from_csv = True
                return cached_df

        # Fetch from database
        df = self.reader.fetch_hourly_collection_plans(self.auto_schedule_ids)

        if df.empty:
            raise RuntimeError("No data returned from the database.")

        # Process the data
        df = self._process_dataset(df)

        # Cache the dataset
        _persist_dataset(df, self.auto_schedule_ids)

        return df

    def _process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and enrich the dataset."""
        df = df.copy()

        # Ensure numeric types for key columns
        numeric_columns = [
            'capacity', 'import_reserved_capacity', 'allocated_capacity',
            'cache_freed_capacity', 'retry_reserved_capacity', 'total_capacity',
            'available_for_utilization', 'sending', 'total_reserved',
            'cache_free_pct', 'import_resv_pct', 'retry_resv_pct',
            'net_utilization_pct', 'hour_utilization_pct', 'plan_hour'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert plan_date to integer format (YYYYMMDD)
        if 'plan_date' in df.columns:
            df['plan_date'] = pd.to_numeric(df['plan_date'], errors='coerce').astype('Int64')

        # Create datetime column for visualization
        # Ensure plan_hour is numeric and handle nulls
        df['plan_hour'] = pd.to_numeric(df['plan_hour'], errors='coerce').fillna(0).astype(int)

        # Build datetime string safely
        df['_datetime_str'] = (
            df['plan_date'].astype(str).str.replace('<NA>', '') +
            df['plan_hour'].astype(str).str.zfill(2)
        )

        # Convert to datetime, filtering out invalid entries
        df['plan_datetime'] = pd.to_datetime(
            df['_datetime_str'],
            format='%Y%m%d%H',
            errors='coerce'
        )
        df = df.drop(columns=['_datetime_str'])

        # Drop rows where datetime conversion failed
        df = df.dropna(subset=['plan_datetime'])

        # Create label for Gantt chart
        df['label'] = (
            df['provider_code'].astype(str) + ' | ' + df['site_code'].astype(str)
        )

        log.info("Processed dataset with %s records", len(df))
        return df


class ComparisonDataLoader:
    """Loader responsible for preparing comparison dashboard data."""

    def __init__(
        self,
        scheduled_reader: HourlyCollectionPlansReader | None = None,
        actual_reader: ActualSentDataReader | None = None,
        auto_schedule_ids: Optional[List[int]] = None,
        sales_date: Optional[int] = None,
        num_ids: int = 2
    ):
        self.scheduled_reader = scheduled_reader or HourlyCollectionPlansReader()
        self.actual_reader = actual_reader or ActualSentDataReader()
        self.auto_schedule_ids = auto_schedule_ids
        self.sales_date = sales_date
        self.num_ids = num_ids
        self._dataset: pd.DataFrame | None = None
        self._loaded_from_csv = False

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Return the prepared comparison dataset, optionally forcing a refresh."""
        if force_refresh or self._dataset is None:
            self._dataset = self._build_comparison_dataset(force_refresh=force_refresh)
        return self._dataset.copy()

    @property
    def loaded_from_cache(self) -> bool:
        """Return True if the last load came from a cached CSV."""
        return self._loaded_from_csv

    def refresh(self) -> pd.DataFrame:
        """Force a rebuild of the underlying dataset and return it."""
        return self.load(force_refresh=True)

    def _build_comparison_dataset(self, force_refresh: bool = False) -> pd.DataFrame:
        log.info("Loading comparison dataset")

        # Get auto_schedule_ids if not provided
        if self.auto_schedule_ids is None:
            self.auto_schedule_ids = self.scheduled_reader.fetch_auto_schedule_ids(limit=self.num_ids)

        if not self.auto_schedule_ids:
            raise RuntimeError("No auto_schedule_ids available")

        # Fetch scheduled data
        scheduled_df = self.scheduled_reader.fetch_hourly_collection_plans(self.auto_schedule_ids)
        if scheduled_df.empty:
            raise RuntimeError("No scheduled data returned from the database.")

        # Process scheduled data
        scheduled_df = self._process_scheduled_data(scheduled_df)

        # Fetch actual sent data if sales_date is provided
        if self.sales_date:
            actual_df = self.actual_reader.fetch_actual_sent_data(self.sales_date)
            if not actual_df.empty:
                actual_df = self._process_actual_data(actual_df)
                # Merge with scheduled data
                comparison_df = self._merge_scheduled_and_actual(scheduled_df, actual_df)
            else:
                log.warning("No actual sent data found for sales_date %s", self.sales_date)
                comparison_df = scheduled_df
                comparison_df['actual_requests'] = 0
                comparison_df['variance'] = comparison_df['sending']
                comparison_df['variance_pct'] = 0.0
        else:
            comparison_df = scheduled_df
            comparison_df['actual_requests'] = 0
            comparison_df['variance'] = comparison_df['sending']
            comparison_df['variance_pct'] = 0.0

        return comparison_df

    def _process_scheduled_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and enrich the scheduled dataset."""
        df = df.copy()

        # Ensure numeric types for key columns
        numeric_columns = [
            'capacity', 'import_reserved_capacity', 'allocated_capacity',
            'cache_freed_capacity', 'retry_reserved_capacity', 'total_capacity',
            'available_for_utilization', 'sending', 'total_reserved',
            'cache_free_pct', 'import_resv_pct', 'retry_resv_pct',
            'net_utilization_pct', 'hour_utilization_pct', 'plan_hour'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert plan_date to integer format (YYYYMMDD)
        if 'plan_date' in df.columns:
            df['plan_date'] = pd.to_numeric(df['plan_date'], errors='coerce').astype('Int64')

        # Create datetime column for visualization
        df['plan_hour'] = pd.to_numeric(df['plan_hour'], errors='coerce').fillna(0).astype(int)

        # Build datetime string safely
        df['_datetime_str'] = (
            df['plan_date'].astype(str).str.replace('<NA>', '') +
            df['plan_hour'].astype(str).str.zfill(2)
        )

        # Convert to datetime
        df['plan_datetime'] = pd.to_datetime(
            df['_datetime_str'],
            format='%Y%m%d%H',
            errors='coerce'
        )
        df = df.drop(columns=['_datetime_str'])

        # Drop rows where datetime conversion failed
        df = df.dropna(subset=['plan_datetime'])

        # Create label for Gantt chart
        df['label'] = (
            df['provider_code'].astype(str) + ' | ' + df['site_code'].astype(str)
        )

        log.info("Processed scheduled dataset with %s records", len(df))
        return df

    def _process_actual_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the actual sent data."""
        df = df.copy()

        # Normalize column names to match scheduled data
        df = df.rename(columns={
            'providercode': 'provider_code',
            'sitecode': 'site_code',
            'scheduledate': 'schedule_date',
            'scheduletime': 'schedule_time',
            'requests': 'actual_requests'
        })

        # Convert date and time to datetime
        # scheduledate format: YYYYMMDD, scheduletime format: HHMM
        df['schedule_date'] = pd.to_numeric(df['schedule_date'], errors='coerce').astype('Int64')
        df['schedule_time'] = pd.to_numeric(df['schedule_time'], errors='coerce').fillna(0).astype(int)

        # Extract hour from schedule_time (HHMM format)
        df['schedule_hour'] = (df['schedule_time'] // 100).astype(int)

        # Build datetime
        df['_datetime_str'] = (
            df['schedule_date'].astype(str).str.replace('<NA>', '') +
            df['schedule_hour'].astype(str).str.zfill(2)
        )

        df['actual_datetime'] = pd.to_datetime(
            df['_datetime_str'],
            format='%Y%m%d%H',
            errors='coerce'
        )
        df = df.drop(columns=['_datetime_str'])

        # Drop rows where datetime conversion failed
        df = df.dropna(subset=['actual_datetime'])

        # Ensure actual_requests is numeric
        df['actual_requests'] = pd.to_numeric(df['actual_requests'], errors='coerce').fillna(0)

        log.info("Processed actual sent dataset with %s records", len(df))
        return df

    def _merge_scheduled_and_actual(self, scheduled_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
        """Merge scheduled and actual data to create comparison dataset."""
        # Aggregate actual data by provider, site, and hour
        actual_agg = actual_df.groupby(['provider_code', 'site_code', 'actual_datetime'], as_index=False).agg({
            'actual_requests': 'sum'
        })

        # Merge with scheduled data
        comparison_df = scheduled_df.merge(
            actual_agg,
            left_on=['provider_code', 'site_code', 'plan_datetime'],
            right_on=['provider_code', 'site_code', 'actual_datetime'],
            how='left'
        )

        # Fill missing actual_requests with 0
        comparison_df['actual_requests'] = comparison_df['actual_requests'].fillna(0)

        # Drop the redundant actual_datetime column
        if 'actual_datetime' in comparison_df.columns:
            comparison_df = comparison_df.drop(columns=['actual_datetime'])

        # Calculate variance (scheduled - actual)
        comparison_df['variance'] = comparison_df['sending'] - comparison_df['actual_requests']

        # Calculate variance percentage
        comparison_df['variance_pct'] = (
            100.0 * comparison_df['variance'] / comparison_df['sending'].replace(0, pd.NA)
        ).fillna(0)

        log.info("Created comparison dataset with %s records", len(comparison_df))
        return comparison_df


__all__ = ["DataLoader", "ComparisonDataLoader", "get_auto_schedule_ids"]
