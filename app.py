"""Entry point for the Hourly Collection Plans Gantt Dashboard."""

import logging
import os
from datetime import datetime
from threading import Lock

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate

from dashboard import (
    ComparisonDataLoader,
    HourlyCollectionPlansReader,
    ActualSentDataReader,
    build_gantt_figure,
)

root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
else:
    root_logger.setLevel(logging.INFO)

log = logging.getLogger(__name__)

# Connection management
_READER_LOCK = Lock()
_scheduled_reader_instance: HourlyCollectionPlansReader | None = None
_actual_reader_instance: ActualSentDataReader | None = None


def _get_scheduled_reader() -> HourlyCollectionPlansReader:
    """Return a shared HourlyCollectionPlansReader instance."""
    global _scheduled_reader_instance
    with _READER_LOCK:
        if _scheduled_reader_instance is None:
            _scheduled_reader_instance = HourlyCollectionPlansReader()
        return _scheduled_reader_instance


def _get_actual_reader() -> ActualSentDataReader:
    """Return a shared ActualSentDataReader instance."""
    global _actual_reader_instance
    with _READER_LOCK:
        if _actual_reader_instance is None:
            _actual_reader_instance = ActualSentDataReader()
        return _actual_reader_instance


def _reset_readers() -> None:
    """Close and recreate the reader connections."""
    global _scheduled_reader_instance, _actual_reader_instance
    with _READER_LOCK:
        if _scheduled_reader_instance is not None:
            try:
                _scheduled_reader_instance.close()
            except Exception:
                pass
        if _actual_reader_instance is not None:
            try:
                _actual_reader_instance.close()
            except Exception:
                pass
        _scheduled_reader_instance = HourlyCollectionPlansReader()
        _actual_reader_instance = ActualSentDataReader()
        log.info("Re-established reader connections")


def _is_connection_error(err: Exception) -> bool:
    """Check if error is a connection issue."""
    if isinstance(err, (BrokenPipeError, ConnectionError)):
        return True
    error_str = str(err).lower()
    return any(
        marker in error_str
        for marker in [
            "eof occurred in violation of protocol",
            "_ssl.c",
            "broken pipe",
            "connection reset",
        ]
    )


# Data storage
_DATASET_LOCK = Lock()
_cached_dataset = None
_cached_ids = None
_cached_date_range: tuple[int | None, int | None] | None = None

# Prevent concurrent load actions (e.g., when the user clicks rapidly).
_LOAD_ACTION_LOCK = Lock()
# Prevent concurrent "Top IDs" requests from stacking when users over-click.
_TOP_IDS_ACTION_LOCK = Lock()

# Shared message for when there are no auto schedule ids in the source table.
_NO_AUTO_SCHEDULE_MSG = "No auto_schedule_id values found in as_hourly_collection_plans"


def _load_dataset_with_retry(auto_schedule_ids: list[int], force_refresh: bool = False):
    """Load dataset with connection retry logic."""
    try:
        return _load_dataset(auto_schedule_ids, force_refresh)
    except Exception as exc:
        if not _is_connection_error(exc):
            raise
        log.warning("Connection error; refreshing readers and retrying.")
        _reset_readers()
        return _load_dataset(auto_schedule_ids, force_refresh)


def _load_dataset(auto_schedule_ids: list[int], force_refresh: bool = False):
    """Load the comparison dataset."""
    global _cached_dataset, _cached_ids, _cached_date_range

    cache_key = tuple(auto_schedule_ids)

    with _DATASET_LOCK:
        if not force_refresh and _cached_ids == cache_key and _cached_dataset is not None:
            log.info("Using cached dataset")
            cached_start, cached_end = (_cached_date_range if _cached_date_range is not None else (None, None))
            return _cached_dataset.copy(), False, cached_start, cached_end

        loader = ComparisonDataLoader(
            scheduled_reader=_get_scheduled_reader(),
            actual_reader=_get_actual_reader(),
            auto_schedule_ids=auto_schedule_ids,
        )
        df = loader.load(force_refresh=force_refresh)
        actual_start_date, actual_end_date = loader.actual_date_range
        _cached_dataset = df.copy()
        _cached_ids = cache_key
        _cached_date_range = (actual_start_date, actual_end_date)
        return df, loader.loaded_from_cache, actual_start_date, actual_end_date


# Initialize Dash app
FILTER_COLUMN_STYLE = {
    'flex': '0 0 220px',
    'minWidth': '180px',
    'maxWidth': '250px',
    'flexShrink': 0,
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Scheduled vs Actual Comparison Dashboard"
server = app.server

# Build layout with filters on left
app.layout = dbc.Container([
    html.H1("Scheduled vs Actual Sending - Comparison Dashboard", className="mb-4"),

    # Control bar
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText("Auto Schedule ID:"),
                dbc.Input(
                    id='auto-schedule-id-input',
                    type='number',
                    min=1,
                    step=1,
                    className="form-control",
                    placeholder="Enter auto_schedule_id"
                ),
            ], size="sm", className="me-2"),
        ], width="auto"),
        dbc.Col([
            dbc.Button("Load Data", id='load-btn', color="primary", size="sm", className="me-2"),
            dbc.Button("Refresh", id='refresh-btn', color="secondary", size="sm", className="me-2"),
            html.Span(id='loading-text', className="ms-2 text-muted"),
        ], width="auto"),
        dbc.Col([
            dbc.Button("Top 10 IDs", id='top-ids-btn', color="info", size="sm", className="me-2"),
        ], width="auto"),
        dbc.Col([
            dbc.Button("Hide Filters", id='toggle-filters-btn', color="secondary", size="sm"),
        ], width="auto"),
    ], className="control-bar align-items-center mb-3"),

    html.Div(id='top-ids-display', className="mb-2 small text-muted"),

    # Status alert
    dbc.Alert(id='status-alert', is_open=False, duration=4000, className="mb-3"),

    # Main content row with filters sidebar and chart
    dbc.Row([
        # Filters sidebar (left) - narrow width
        dbc.Col([
            dbc.Collapse([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Filters", className="mb-3"),

                        html.Div([
                            html.Label("Date Range", className="mb-1", style={'fontSize': '0.85rem'}),
                            dcc.DatePickerRange(
                                id='filter-date-range',
                                display_format='MMM DD',
                                start_date_placeholder_text='Start',
                                end_date_placeholder_text='End',
                                style={'width': '100%', 'fontSize': '0.75rem'},
                                className="mb-3"
                            ),
                        ]),

                        html.Div([
                            html.Label("Provider", className="mb-1", style={'fontSize': '0.85rem'}),
                            dcc.Dropdown(
                                id='filter-provider',
                                options=[],
                                value='all',
                                multi=True,
                                placeholder="All",
                                className="mb-3",
                                style={'fontSize': '0.8rem'}
                            ),
                        ]),

                        html.Div([
                            html.Label("Site", className="mb-1", style={'fontSize': '0.85rem'}),
                            dcc.Dropdown(
                                id='filter-site',
                                options=[],
                                value='all',
                                multi=True,
                                placeholder="All",
                                className="mb-3",
                                style={'fontSize': '0.8rem'}
                            ),
                        ]),

                        html.Div([
                            html.Label("Schedule ID", className="mb-1", style={'fontSize': '0.85rem'}),
                            dcc.Dropdown(
                                id='filter-auto-schedule-id',
                                options=[],
                                value='all',
                                multi=True,
                                placeholder="All",
                                className="mb-3",
                                style={'fontSize': '0.8rem'}
                            ),
                        ]),

                        html.Div([
                            html.Label("Show Source Types", className="mb-1", style={'fontSize': '0.85rem'}),
                            dcc.Checklist(
                                id='filter-source-type',
                                options=[
                                    {'label': 'Scheduled & Actual', 'value': 'Scheduled & Actual'},
                                    {'label': 'Scheduled Only', 'value': 'Scheduled Only'},
                                    {'label': 'Actual Only', 'value': 'Actual Only'},
                                ],
                                value=['Scheduled & Actual', 'Scheduled Only', 'Actual Only'],
                                labelStyle={'display': 'block', 'fontSize': '0.75rem', 'marginBottom': '0.2rem'},
                                inputStyle={'marginRight': '0.5rem'}
                            ),
                        ]),

                        html.Div([
                            html.Label("Color By", className="mb-1", style={'fontSize': '0.85rem'}),
                            dcc.Dropdown(
                                id='color-field-dropdown',
                                options=[
                                    {'label': 'Request Source (Scheduled vs Actual)', 'value': 'source_type'},
                                    {'label': 'Difference (Actual - Scheduled)', 'value': 'difference'},
                                    {'label': 'Difference %', 'value': 'difference_pct'},
                                    {'label': 'Actual Requests', 'value': 'actual_requests'},
                                    {'label': 'Scheduled Requests', 'value': 'sending'},
                                    {'label': 'Capacity', 'value': 'total_capacity'},
                                ],
                                value='source_type',
                                className="mb-3",
                                style={'fontSize': '0.8rem'}
                            ),
                        ]),
                    ], style={'padding': '0.75rem'})
                ], style={'minWidth': '180px', 'maxWidth': '200px'})
            ], id='filters-collapse', is_open=True)
        ], id='filters-column', width='auto', className="filters-sidebar", style=dict(FILTER_COLUMN_STYLE)),

        # Main chart area (right)
        dbc.Col([
            dcc.Loading(
                id='gantt-loading',
                type='default',
                children=dcc.Graph(id='gantt-chart', config={'displayModeBar': True})
            )
        ], width=True, className="main-content")
    ], className="dashboard-body", style={'flexWrap': 'nowrap'}),

    # Hidden stores
    dcc.Store(id='dataset-store'),
    dcc.Store(id='load-complete'),

], fluid=True, className="py-3")


@app.callback(
    [Output('dataset-store', 'data'),
     Output('status-alert', 'children'),
     Output('status-alert', 'color'),
     Output('status-alert', 'is_open'),
     Output('load-complete', 'data')],
    [Input('load-btn', 'n_clicks'),
     Input('refresh-btn', 'n_clicks')],
    [State('auto-schedule-id-input', 'value')]
)
def load_data(load_clicks, refresh_clicks, auto_schedule_id):
    """Load or refresh the comparison dataset for a single auto_schedule_id."""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id not in ('load-btn', 'refresh-btn'):
        raise PreventUpdate

    force_refresh = triggered_id == 'refresh-btn'

    lock_acquired = False
    try:
        if not auto_schedule_id:
            return (
                no_update,
                "Please enter an auto_schedule_id to load data",
                'warning',
                True,
                datetime.now().isoformat()
            )

        try:
            auto_schedule_id_int = int(auto_schedule_id)
        except (TypeError, ValueError):
            return (
                no_update,
                "Invalid auto_schedule_id. Please enter a valid number.",
                'warning',
                True,
                datetime.now().isoformat()
            )

        if auto_schedule_id_int < 1:
            return (
                no_update,
                "auto_schedule_id must be a positive integer.",
                'warning',
                True,
                datetime.now().isoformat()
            )

        lock_acquired = _LOAD_ACTION_LOCK.acquire(blocking=False)
        if not lock_acquired:
            log.info("Load already in progress; ignoring extra click.")
            raise PreventUpdate

        auto_schedule_ids = [auto_schedule_id_int]

        # Load the comparison dataset
        df, loaded_from_cache, actual_start_date, actual_end_date = _load_dataset_with_retry(auto_schedule_ids, force_refresh)

        record_count = len(df)
        message = f"Loaded {record_count:,} records for auto_schedule_id: {auto_schedule_id_int}"
        if actual_start_date:
            if actual_end_date and actual_end_date != actual_start_date:
                message += f" | Sales Date Range: {actual_start_date} - {actual_end_date}"
            else:
                message += f" | Sales Date: {actual_start_date}"
        if loaded_from_cache:
            message += " (from cache)"

        return (
            {
                'auto_schedule_ids': auto_schedule_ids,
                'start_date': actual_start_date,
                'end_date': actual_end_date,
                'loaded_at': datetime.now().isoformat()
            },
            message,
            'success',
            True,
            datetime.now().isoformat()
        )

    except PreventUpdate:
        raise
    except Exception as exc:
        error_msg = str(exc)
        if _NO_AUTO_SCHEDULE_MSG in error_msg:
            _reset_readers()
        log.error("Error loading data: %s", error_msg)
        return (
            no_update,
            f"Error loading data: {error_msg}",
            'danger',
            True,
            datetime.now().isoformat()
        )
    finally:
        if lock_acquired:
            _LOAD_ACTION_LOCK.release()


@app.callback(
    Output('top-ids-display', 'children'),
    Input('top-ids-btn', 'n_clicks')
)
def display_top_ids(n_clicks):
    """Show the latest top N auto_schedule_ids from the scheduled plan table."""
    if not n_clicks:
        raise PreventUpdate

    lock_acquired = _TOP_IDS_ACTION_LOCK.acquire(blocking=False)
    if not lock_acquired:
        log.info("Top IDs request already in progress; ignoring extra click.")
        raise PreventUpdate

    try:
        reader = _get_scheduled_reader()
        ids = reader.fetch_auto_schedule_ids(limit=10)
    except Exception as exc:
        log.error("Unable to fetch top auto_schedule_ids: %s", exc)
        return f"Error fetching top IDs: {exc}"
    finally:
        _TOP_IDS_ACTION_LOCK.release()

    if not ids:
        return "No auto_schedule_ids available."

    ids_list = ', '.join(str(value) for value in ids)
    return f"Top 10 auto_schedule_ids: {ids_list}"


@app.callback(
    [Output('filters-collapse', 'is_open'),
     Output('toggle-filters-btn', 'children'),
     Output('filters-column', 'style')],
    Input('toggle-filters-btn', 'n_clicks'),
    State('filters-collapse', 'is_open')
)
def toggle_filters(n_clicks, is_open):
    """Toggle filters sidebar visibility."""
    if is_open is None:
        is_open = True

    base_style = dict(FILTER_COLUMN_STYLE)
    hidden_style = {
        **FILTER_COLUMN_STYLE,
        'flex': '0 0 0',
        'minWidth': '0',
        'maxWidth': '0',
        'width': '0',
        'overflow': 'hidden',
        'padding': '0',
        'margin': '0',
    }

    if not n_clicks:
        label = "Hide Filters" if is_open else "Show Filters"
        style = base_style if is_open else hidden_style
        return is_open, label, style

    new_state = not is_open
    label = "Hide Filters" if new_state else "Show Filters"
    style = base_style if new_state else hidden_style
    return new_state, label, style


@app.callback(
    [Output('filter-provider', 'options'),
     Output('filter-site', 'options'),
     Output('filter-auto-schedule-id', 'options')],
    Input('dataset-store', 'data')
)
def update_filter_options(dataset_data):
    """Update filter dropdown options when data is loaded."""
    if not dataset_data or 'auto_schedule_ids' not in dataset_data:
        return [[{'label': 'All', 'value': 'all'}]] * 3

    auto_schedule_ids = dataset_data['auto_schedule_ids']
    cache_key = tuple(auto_schedule_ids)

    # Get the cached dataset
    with _DATASET_LOCK:
        if _cached_ids != cache_key or _cached_dataset is None:
            return [[{'label': 'All', 'value': 'all'}]] * 3
        df = _cached_dataset.copy()

    # Provider options
    provider_options = [{'label': 'All', 'value': 'all'}]
    providers = df['provider_code'].dropna().unique()
    provider_options.extend({'label': p, 'value': p} for p in sorted(providers))

    # Site options
    site_options = [{'label': 'All', 'value': 'all'}]
    sites = df['site_code'].dropna().unique()
    site_options.extend({'label': s, 'value': s} for s in sorted(sites))

    # Auto Schedule ID options
    id_options = [{'label': 'All', 'value': 'all'}]
    ids = df['auto_schedule_id'].dropna().unique()
    id_options.extend({'label': str(i), 'value': i} for i in sorted(ids, reverse=True))

    return provider_options, site_options, id_options


@app.callback(
    [Output('loading-text', 'children'),
     Output('load-btn', 'disabled'),
     Output('refresh-btn', 'disabled')],
    [Input('load-btn', 'n_clicks'),
     Input('refresh-btn', 'n_clicks'),
     Input('load-complete', 'data')]
)
def loading_state(load_clicks, refresh_clicks, load_complete):
    """Show loading indicator and disable buttons during load."""
    ctx = callback_context
    if not ctx.triggered:
        return "", False, False

    triggered_ids = {t['prop_id'].split('.')[0] for t in ctx.triggered if t.get('prop_id')}

    if 'load-complete' in triggered_ids:
        return "", False, False
    elif triggered_ids.intersection({'load-btn', 'refresh-btn'}):
        return "Loading...", True, True

    return "", False, False


@app.callback(
    Output('gantt-chart', 'figure'),
    [Input('dataset-store', 'data'),
     Input('color-field-dropdown', 'value'),
     Input('filter-provider', 'value'),
     Input('filter-site', 'value'),
     Input('filter-auto-schedule-id', 'value'),
    Input('filter-source-type', 'value'),
     Input('filter-date-range', 'start_date'),
     Input('filter-date-range', 'end_date')]
)
def update_gantt(dataset_data, color_field, provider_filter, site_filter, id_filter, source_filter, start_date, end_date):
    """Update the Gantt chart when data or filters change."""
    if not dataset_data or 'auto_schedule_ids' not in dataset_data:
        raise PreventUpdate

    auto_schedule_ids = dataset_data['auto_schedule_ids']
    cache_key = tuple(auto_schedule_ids)

    # Get the cached dataset
    with _DATASET_LOCK:
        if _cached_ids != cache_key or _cached_dataset is None:
            raise PreventUpdate
        df = _cached_dataset.copy()

    # Apply filters
    if provider_filter and 'all' not in provider_filter:
        if isinstance(provider_filter, list):
            df = df[df['provider_code'].isin(provider_filter)]
        else:
            df = df[df['provider_code'] == provider_filter]

    if site_filter and 'all' not in site_filter:
        if isinstance(site_filter, list):
            df = df[df['site_code'].isin(site_filter)]
        else:
            df = df[df['site_code'] == site_filter]

    if id_filter and 'all' not in id_filter:
        if isinstance(id_filter, list):
            df = df[df['auto_schedule_id'].isin(id_filter)]
        else:
            df = df[df['auto_schedule_id'] == id_filter]

    valid_sources = {'Scheduled & Actual', 'Scheduled Only', 'Actual Only'}
    if source_filter:
        if isinstance(source_filter, str):
            selected = [source_filter]
        else:
            selected = list(source_filter)
        selected = [value for value in selected if value in valid_sources]
        if selected:
            df = df[df['source_type'].isin(selected)]

    # Apply date picker filters (only affects the visualization, not the loaded data)
    if start_date:
        start_dt = pd.to_datetime(start_date, errors='coerce')
        if pd.notna(start_dt):
            df = df[df['plan_datetime'] >= start_dt]

    if end_date:
        end_dt = pd.to_datetime(end_date, errors='coerce')
        if pd.notna(end_dt):
            end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df['plan_datetime'] <= end_dt]

    # Build the Gantt figure
    fig = build_gantt_figure(df, color_field=color_field or 'source_type')

    return fig


if __name__ == '__main__':
    app.run(debug=True, port=8051)
