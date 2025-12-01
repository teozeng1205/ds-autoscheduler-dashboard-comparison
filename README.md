# Scheduled vs Actual Comparison Dashboard

A simplified Dash dashboard for comparing scheduled hourly collection plans (what we plan to send) vs actual sent data (what was actually sent) from provider audit logs.

## Purpose

This dashboard compares:
- **Scheduled Requests**: Hourly collection plans from PriceEye `as_hourly_collection_plans` table (what we are scheduled to send)
- **Actual Requests**: Provider audit data from `prod.monitoring.provider_combined_audit` table (what was actually sent)

The comparison helps identify differences between planned and actual provider file sends, enabling better monitoring and scheduling optimization.

## Features

- **Simplified Comparison View**: Focus on key metrics only
  - **Capacity**: Total capacity available
  - **Scheduled Requests**: Number of requests we planned to send
  - **Actual Requests**: Number of requests actually sent
  - **Difference**: Actual - Scheduled (positive = sent more than planned, negative = sent less)
  - **Difference %**: Percentage difference relative to scheduled

- **Gantt Chart**: Interactive timeline visualization showing provider/site data over time

- **Multiple Color Options**: Visualize by different comparison metrics:
  - Difference (Actual - Scheduled)
  - Difference %
  - Actual Requests
  - Scheduled Requests
  - Capacity

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
  WHERE sales_date BETWEEN <START_DATE> AND <END_DATE>
  GROUP BY 1, 2, 3, 4
  ```
- **Range Source**: `<START_DATE>` and `<END_DATE>` are derived from the earliest and latest plan datetimes for the selected `auto_schedule_id` in `as_hourly_collection_plans`.

## Running the Dashboard

```bash
python app.py
```

The dashboard will be available at: http://127.0.0.1:8051/

## Usage

1. **Enter Auto Schedule ID**: Provide the `auto_schedule_id` you want to analyze—only a single ID is required.
2. **Load Data**: Click "Load Data" to fetch the scheduled plans and automatically derived actual data.
   - The sales date range is taken from the earliest and latest scheduled plan datetimes for that `auto_schedule_id`, so you no longer need to enter manual start/end dates.
3. **Top 10 IDs**: Click the "Top 10 IDs" button to see a quick list of the most recent `auto_schedule_id` values in the control bar.
4. **Refresh**: Click "Refresh" to reload from the database (bypassing cache)
5. **Select Color Metric**: Choose which comparison metric to visualize in the Gantt chart
6. **Apply Filters**: Use the sidebar filters to narrow down by provider, site, schedule ID, or date range

## Hover Information

When hovering over bars in the Gantt chart, you'll see:

- **Provider/Site**
- **Time** (hour)
- **Capacity**: Total capacity available
- **Scheduled Requests**: Number of requests planned to be sent
- **Actual Requests**: Number of requests actually sent
- **Difference**: Actual - Scheduled
- **Difference %**: Percentage difference

## Understanding the Difference

- **Positive Difference**: We sent more than scheduled (over-delivery)
- **Negative Difference**: We sent less than scheduled (under-delivery)
- **Zero Difference**: Perfect match between scheduled and actual

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
