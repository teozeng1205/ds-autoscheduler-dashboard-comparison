"""Database helpers for retrieving hourly collection plans data."""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
from threevictors.dao import redshift_connector


log = logging.getLogger(__name__)


class HourlyCollectionPlansReader(redshift_connector.RedshiftConnector):
    """Reader that pulls hourly collection plans data from PriceEye."""

    def __init__(self):
        log.info("Initializing HourlyCollectionPlansReader")
        super().__init__()
        log.info("HourlyCollectionPlansReader ready")

    def get_properties_filename(self) -> str:
        """Return the credential properties filename for ThreeVictors."""
        return "database-core-local-redshift-serverless-reader.properties"

    def fetch_auto_schedule_ids(self, limit: int = 10) -> List[int]:
        """Return the most recent auto_schedule_id values."""
        query = """
            SELECT DISTINCT auto_schedule_id
            FROM local.federated_priceeye.as_hourly_collection_plans
            ORDER BY 1 DESC
            LIMIT %s
        """
        with self.get_connection().cursor() as cursor:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

        if not rows:
            raise RuntimeError("No auto_schedule_id values found in as_hourly_collection_plans")

        ids = [int(row[0]) for row in rows]
        log.info("Found %s auto_schedule_ids: %s", len(ids), ids)
        return ids

    def fetch_hourly_collection_plans(self, auto_schedule_ids: List[int]) -> pd.DataFrame:
        """Return the hourly collection plans data for the specified auto_schedule_ids."""
        if not auto_schedule_ids:
            return pd.DataFrame()

        # Build the IDs list for the query
        ids_list = ','.join(str(id) for id in auto_schedule_ids)

        query = f"""
WITH ids AS (
  SELECT DISTINCT auto_schedule_id
  FROM local.federated_priceeye.as_hourly_collection_plans
  WHERE auto_schedule_id IN ({ids_list})
  ORDER BY 1 DESC
),
base AS (
  SELECT
    hcp.*,
    (hcp.capacity + hcp.cache_freed_capacity)                                  AS total_capacity,
    (hcp.capacity + hcp.cache_freed_capacity
       - (hcp.import_reserved_capacity + hcp.retry_reserved_capacity))         AS available_for_utilization,
    (hcp.allocated_capacity - hcp.import_reserved_capacity)                    AS sending,
    (hcp.allocated_capacity - hcp.import_reserved_capacity
       + hcp.import_reserved_capacity + hcp.retry_reserved_capacity)           AS total_reserved
  FROM local.federated_priceeye.as_hourly_collection_plans hcp
  JOIN ids ON hcp.auto_schedule_id = ids.auto_schedule_id
)
SELECT
  auto_schedule_id,
  id,
  provider_code,
  site_code,
  plan_date,
  plan_hour,
  capacity,
  import_reserved_capacity,
  allocated_capacity,
  cache_freed_capacity,
  retry_reserved_capacity,

  -- derived totals
  total_capacity,
  available_for_utilization,
  sending,
  total_reserved,
  100.0 * cache_freed_capacity        / NULLIF(capacity, 0) AS cache_free_pct,
  100.0 * import_reserved_capacity    / NULLIF(capacity, 0) AS import_resv_pct,
  100.0 * retry_reserved_capacity     / NULLIF(capacity, 0) AS retry_resv_pct,
  100.0 * allocated_capacity          / NULLIF(available_for_utilization, 0) AS net_utilization_pct,
  100.0 * total_reserved              / NULLIF(total_capacity, 0) AS hour_utilization_pct
FROM base
ORDER BY auto_schedule_id DESC, plan_date, plan_hour
        """

        try:
            with self.get_connection().cursor() as cursor:
                cursor.execute(query)
                colnames = [desc[0] for desc in cursor.description]
                records = cursor.fetchall()
        except Exception as e:
            log.error("Error fetching hourly collection plans: %s", e)
            raise

        df = pd.DataFrame(records, columns=colnames)
        log.info("Fetched %s rows for auto_schedule_ids %s", len(df), auto_schedule_ids)
        return df


class ActualSentDataReader(redshift_connector.RedshiftConnector):
    """Reader that pulls actual sent data from the analytics database."""

    def __init__(self):
        log.info("Initializing ActualSentDataReader")
        super().__init__()
        log.info("ActualSentDataReader ready")

    def get_properties_filename(self) -> str:
        """Return the credential properties filename for ThreeVictors."""
        return "database-analytics-redshift-serverless-reader.properties"

    def fetch_actual_sent_data(self, start_date: int, end_date: int | None = None) -> pd.DataFrame:
        """
        Fetch actual sent data from provider_combined_audit table.

        Args:
            start_date: Start sales date in YYYYMMDD format (e.g., 20251127)
            end_date: End sales date in YYYYMMDD format (optional, defaults to start_date)

        Returns:
            DataFrame with columns: providercode, sitecode, scheduledate, scheduletime, requests
        """
        if end_date is None:
            end_date = start_date

        query = """
            SELECT providercode, sitecode, scheduledate, scheduletime, count(*) as requests
            FROM prod.monitoring.provider_combined_audit
            WHERE sales_date BETWEEN %s AND %s
            GROUP BY 1, 2, 3, 4
        """

        try:
            with self.get_connection().cursor() as cursor:
                cursor.execute(query, (start_date, end_date))
                colnames = [desc[0] for desc in cursor.description]
                records = cursor.fetchall()
        except Exception as e:
            log.error("Error fetching actual sent data: %s", e)
            raise

        df = pd.DataFrame(records, columns=colnames)
        log.info("Fetched %s rows for sales_date range %s to %s", len(df), start_date, end_date)
        return df


__all__ = ["HourlyCollectionPlansReader", "ActualSentDataReader"]
