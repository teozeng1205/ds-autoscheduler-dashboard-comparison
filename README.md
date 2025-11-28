# Scheduled vs Actual Comparison Dashboard

A Dash dashboard for comparing scheduled hourly collection plans (what we plan to send) vs actual sent data (what was actually sent) from provider audit logs.

## Purpose

This dashboard compares:
- **Scheduled Data**: Hourly collection plans from PriceEye `as_hourly_collection_plans` table (what we are scheduled to send)
- **Actual Sent Data**: Provider audit data from `prod.monitoring.provider_combined_audit` table (what was actually sent)

The comparison helps identify variances between planned and actual provider file sends, enabling better monitoring and scheduling optimization.

## Features

- **Comparison View**: Side-by-side comparison of scheduled vs actual sent requests
- **Gantt Chart**: Interactive timeline visualization showing provider/site data over time
- **Variance Metrics**: Automatically calculates:
  - **Variance**: Difference between scheduled and actual (scheduled - actual)
  - **Variance %**: Percentage variance relative to scheduled
- **Multiple Color Options**: Visualize by different comparison metrics:
  - Variance (absolute difference)
  - Variance % (percentage difference)
  - Actual Requests (what was actually sent)
  - Scheduled/Sending (what was planned to be sent)
  - Hour Utilization %
  - Net Utilization %
  - Allocated Capacity
  - Total Capacity
- **Flexible Filters**: Filter by provider, site, schedule ID, and date range

## Data Sources

### Scheduled Data
- **Database**: `database-core-local-redshift-serverless-reader.properties`
- **Table**: `local.federated_priceeye.as_hourly_collection_plans`
- **Key Metrics**: Capacity, allocated capacity, sending (allocated - import reserved)

### Actual Sent Data
- **Database**: `database-analytics-redshift-serverless-reader.properties`
- **Table**: `prod.monitoring.provider_combined_audit`
- **Query**:
  ```sql
  SELECT providercode, sitecode, scheduledate, scheduletime, count(*) as requests
  FROM prod.monitoring.provider_combined_audit
  WHERE sales_date = <YYYYMMDD>
  GROUP BY 1, 2, 3, 4
  ```

## Running the Dashboard

```bash
python app.py
```

The dashboard will be available at: http://127.0.0.1:8051/

## Usage

1. **Set Number of Schedule IDs**: Choose how many recent auto_schedule_ids to load (1-10)
2. **Enter Sales Date**: Input the sales date in YYYYMMDD format (e.g., 20251127) to load actual sent data for comparison
   - If no sales date is provided, only scheduled data will be shown
3. **Load Data**: Click "Load Data" to fetch both scheduled and actual data
4. **Refresh**: Click "Refresh" to reload from the database (bypassing cache)
5. **Select Color Metric**: Choose which comparison metric to visualize in the Gantt chart
6. **Apply Filters**: Use the sidebar filters to narrow down by provider, site, schedule ID, or date range

## Hover Information

When hovering over bars in the Gantt chart, you'll see:

- Provider/Site
- Time (hour)
- Capacity details (base, total, available)
- Allocation details (allocated, sending, total reserved)
- Reserved breakdown (import, retry, cache freed)
- Utilization percentages
- **Comparison metrics** (when actual data is loaded):
  - Scheduled (Sending)
  - Actual Sent
  - Variance
  - Variance %

## Understanding the Comparison

- **Positive Variance**: We scheduled more than what was actually sent (over-scheduled)
- **Negative Variance**: We scheduled less than what was actually sent (under-scheduled)
- **Zero Variance**: Perfect match between scheduled and actual

## Structure

```
ds-autoscheduler-dashboard-comparison/
├── app.py                  # Main application entry point
├── dashboard/              # Dashboard module
│   ├── __init__.py        # Package exports
│   ├── db.py              # Database readers (scheduled & actual)
│   ├── data.py            # Data loading and comparison logic
│   └── figures.py         # Gantt chart visualization
└── resources/             # Cache directory
    └── cache/             # Cached query results
```
