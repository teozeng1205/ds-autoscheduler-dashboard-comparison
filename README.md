# Hourly Collection Plans Dashboard

A simplified Dash dashboard for visualizing hourly collection plans from the PriceEye `as_hourly_collection_plans` table.

## Features

- **Single View**: Focused view of hourly collection plans data
- **Gantt Chart**: Interactive timeline visualization showing provider/site utilization over time
- **Double-Axis Time Display**: Hours on primary axis, dates on secondary axis (same format as original dashboard)
- **Custom Query**: Uses the specified query to fetch and process hourly collection plan data
- **Multiple Color Options**: Visualize by different metrics:
  - Hour Utilization %
  - Net Utilization %
  - Allocated Capacity
  - Total Capacity
  - Sending

## Data Source

The dashboard queries `local.federated_priceeye.as_hourly_collection_plans` and calculates:

- **Total Capacity**: Base capacity + cache freed capacity
- **Available for Utilization**: Total capacity - reserved capacities
- **Sending**: Allocated capacity - import reserved capacity
- **Total Reserved**: All reserved capacity
- **Various Utilization Percentages**: Cache free %, import reserved %, retry reserved %, net utilization %, hour utilization %

## Running the Dashboard

```bash
python app.py
```

The dashboard will be available at: http://127.0.0.1:8051/

## Usage

1. **Set Number of Auto Schedule IDs**: Choose how many recent auto_schedule_ids to load (1-10)
2. **Select Color Metric**: Choose which metric to visualize in the Gantt chart
3. **Load Data**: Click "Load Data" to fetch the data
4. **Refresh**: Click "Refresh" to reload from the database (bypassing cache)

## Hover Information

When hovering over bars in the Gantt chart, you'll see:

- Provider/Site
- Time (hour)
- Capacity details (base, total, available)
- Allocation details (allocated, sending, total reserved)
- Reserved breakdown (import, retry, cache freed)
- Utilization percentages

## Caching

The dashboard caches query results to improve performance. Data is cached in `resources/cache/` directory.

## Structure

```
ds-autoscheduler-dashboard/
├── app.py                  # Main application entry point
├── dashboard/              # Dashboard module
│   ├── __init__.py        # Package exports
│   ├── db.py              # Database reader
│   ├── data.py            # Data loading and caching
│   └── figures.py         # Gantt chart visualization
└── resources/             # Cache directory
    └── cache/             # Cached query results
```
