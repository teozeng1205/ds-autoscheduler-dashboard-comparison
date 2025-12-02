"""Plotly figure builders for the hourly collection plans dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DEFAULT_FIGURE_HEIGHT = 1620  # Base height for reference
MIN_FIGURE_HEIGHT = 400  # Minimum height
MAX_FIGURE_HEIGHT = 3000  # Maximum height
ROW_HEIGHT = 18  # Height per row - more compact
BASE_MARGIN = 270  # Space for margins, title, legend, etc.


def _calculate_dynamic_height(num_rows: int) -> int:
    """Calculate figure height based on number of rows."""
    calculated_height = (num_rows * ROW_HEIGHT) + BASE_MARGIN
    return max(MIN_FIGURE_HEIGHT, min(calculated_height, MAX_FIGURE_HEIGHT))


def _empty_figure() -> go.Figure:
    """Return an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        title="No data available",
        xaxis_title="Time",
        yaxis_title="Provider | Site",
        height=MIN_FIGURE_HEIGHT,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return fig


def build_gantt_figure(df: pd.DataFrame, color_field: str = 'hour_utilization_pct') -> go.Figure:
    """
    Build a Gantt chart showing hourly collection plans.

    Args:
        df: DataFrame with hourly collection plan data
        color_field: Field to use for coloring bars (default: hour_utilization_pct)

    Returns:
        Plotly figure object
    """
    if df.empty or 'plan_datetime' not in df.columns:
        return _empty_figure()

    # Prepare data for visualization
    plot_df = df.copy()

    # Ensure we have necessary columns
    required_cols = ['plan_datetime', 'label']
    if not all(col in plot_df.columns for col in required_cols):
        return _empty_figure()

    # Ensure plan_datetime is actually a datetime type
    plot_df['plan_datetime'] = pd.to_datetime(plot_df['plan_datetime'], errors='coerce')

    # Drop any rows where datetime is invalid
    plot_df = plot_df.dropna(subset=['plan_datetime'])

    if plot_df.empty:
        return _empty_figure()

    # Create end datetime (1 hour after start)
    plot_df['end_datetime'] = plot_df['plan_datetime'] + pd.Timedelta(hours=1)

    # Ensure color field exists
    if color_field not in plot_df.columns:
        color_field = 'allocated_capacity'

    if color_field == 'source_type':
        plot_df[color_field] = plot_df[color_field].astype(str)
    else:
        plot_df[color_field] = pd.to_numeric(plot_df[color_field], errors='coerce').fillna(0)

    # Sort by label and datetime
    plot_df = plot_df.sort_values(['label', 'plan_datetime'])

    # Calculate dynamic height based on number of unique labels (rows)
    num_unique_labels = plot_df['label'].nunique()
    dynamic_height = _calculate_dynamic_height(num_unique_labels)

    color_label = color_field.replace('_', ' ').title()

    color_series = plot_df[color_field]
    is_categorical = (
        not pd.api.types.is_numeric_dtype(color_series)
        or color_series.nunique(dropna=True) <= 5
    )

    if color_field == 'source_type':
        color_label = "Request Source"
        color_discrete_map = {
            'Planned & Actual': '#2ca02c',
            'Planned Only': '#1f77b4',
            'Actual Only': '#d62728',
        }
    else:
        color_discrete_map = None

    # Build custom data for hover - simplified metrics only
    custom_data_fields = [
        'auto_schedule_id', 'provider_code', 'site_code', 'plan_date', 'plan_hour',
        'total_capacity', 'sending', 'actual_requests', 'difference', 'difference_pct'
    ]
    custom_data = [col for col in custom_data_fields if col in plot_df.columns]

    title = (
        "Hourly Collection Plans - Provider/Site Timeline "
        f"(Color by: {color_label}; Actual - Planned)"
    )

    color_kwargs = dict(
        data_frame=plot_df,
        x_start='plan_datetime',
        x_end='end_datetime',
        y='label',
        color=color_field,
        custom_data=custom_data,
        title=title
    )

    if is_categorical and color_discrete_map:
        color_kwargs['color_discrete_map'] = color_discrete_map
    else:
        if color_field in {'difference', 'difference_pct'}:
            color_kwargs['color_continuous_scale'] = 'RdBu'
            color_kwargs['color_continuous_midpoint'] = 0
        else:
            color_kwargs['color_continuous_scale'] = (
                'RdYlGn_r' if 'utilization' in color_field or 'pct' in color_field else 'Blues'
            )
    legend_title = color_label

    fig = px.timeline(**color_kwargs)

    # Build hover template - simplified
    hover_parts = [
        "<b>%{y}</b><br>",
        "Time: %{x|%Y-%m-%d %H:00}<br>",
        "<br>",
    ]

    # Capacity
    if 'total_capacity' in custom_data:
        hover_parts.append(f"<b>Capacity:</b> %{{customdata[{custom_data.index('total_capacity')}]:,.0f}}<br>")

    # Requests comparison
    if 'sending' in custom_data:
        hover_parts.append(f"<b>Planned Requests:</b> %{{customdata[{custom_data.index('sending')}]:,.0f}}<br>")

    if 'actual_requests' in custom_data:
        hover_parts.append(f"<b>Actual Requests:</b> %{{customdata[{custom_data.index('actual_requests')}]:,.0f}}<br>")

    # Difference
    if 'difference' in custom_data:
        hover_parts.append(f"<b>Difference:</b> %{{customdata[{custom_data.index('difference')}]:,.0f}}<br>")

    if 'difference_pct' in custom_data:
        hover_parts.append(f"<b>Difference %:</b> %{{customdata[{custom_data.index('difference_pct')}]:.1f}}%<br>")

    if 'auto_schedule_id' in custom_data:
        hover_parts.append(f"<br>Schedule ID: %{{customdata[{custom_data.index('auto_schedule_id')}]}}<br>")

    hover_template = ''.join(hover_parts) + "<extra></extra>"

    # Update layout and styling
    layout_kwargs = dict(
        xaxis=dict(
            title="",
            domain=[0, 1]
        ),
        yaxis=dict(
            title="Provider | Site",
            automargin=True,
            tickfont=dict(size=10),
            tickmode='linear',
            dtick=1
        ),
        hovermode='closest',
        height=dynamic_height,  # Use dynamically calculated height
        margin=dict(l=150, r=40, t=70, b=200),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.05,  # Reduce gap between bars (default is 0.2)
    )

    if not (is_categorical and color_discrete_map):
        layout_kwargs['coloraxis_colorbar_title'] = legend_title

    fig.update_layout(**layout_kwargs)

    if is_categorical and color_discrete_map:
        fig.update_layout(legend_title_text=legend_title, showlegend=True)

    # Update traces
    fig.update_traces(
        hovertemplate=hover_template,
        marker_line_color="rgba(0,0,0,0.1)",
        marker_line_width=0.5
    )

    # Reverse y-axis to show first items at top
    fig.update_yaxes(autorange="reversed")

    # Configure x-axis with hour ticks
    min_start = plot_df['plan_datetime'].min()
    start_hour = min_start.floor('h') if pd.notna(min_start) else None

    xaxis_kwargs = dict(
        tickformat="%H:%M",
        dtick=14400000,  # 4 hours in milliseconds
        ticklabelmode="instant",
        tickangle=-45,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        ticks="outside",
        ticklen=6,
        tickfont=dict(size=11),
        showline=True,
        linecolor="rgba(0,0,0,0.2)",
        ticklabelposition="outside bottom"
    )
    if start_hour is not None:
        xaxis_kwargs['tick0'] = start_hour

    fig.update_xaxes(**xaxis_kwargs)

    # Add date labels on second line
    if start_hour is not None:
        max_end = plot_df['end_datetime'].max()
        if pd.notna(max_end):
            tick_range = pd.date_range(start=start_hour, end=max_end.ceil('h'), freq='4h')
            if not tick_range.empty:
                tick_text = []
                prev_day = None
                nbsp = '\u00A0'  # Non-breaking space
                for tick in tick_range:
                    time_str = tick.strftime('%H:%M')
                    day_str = tick.strftime('%b %d') if prev_day != tick.date() else nbsp
                    tick_text.append(f"{time_str}<br><br><br>{day_str}")
                    prev_day = tick.date()
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=tick_range,
                    ticktext=tick_text
                )

    return fig


__all__ = ["build_gantt_figure"]
