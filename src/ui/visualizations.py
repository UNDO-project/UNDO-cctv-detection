"""Create interactive visualizations for model metrics.

This module provides Plotly-based visualization functions for training
metrics, performance comparisons, and benchmarking results.
"""

from typing import Any

import plotly.graph_objects as go


def create_loss_curve(metrics: dict[str, Any]) -> go.Figure:
    """Create loss curve visualization for training history.

    :param metrics: Metrics dictionary containing train_loss and val_loss
    :return: Plotly figure with training and validation loss curves
    :rtype: go.Figure
    """
    fig = go.Figure()

    # Add training loss trace
    fig.add_trace(
        go.Scatter(
            y=metrics["train_loss"],
            mode="lines",
            name="Training Loss",
            line={"color": "blue", "width": 2},
            hovertemplate="Epoch %{x}<br>Train Loss: %{y:.3f}<extra></extra>",
        )
    )

    # Add validation loss trace
    fig.add_trace(
        go.Scatter(
            y=metrics["val_loss"],
            mode="lines",
            name="Validation Loss",
            line={"color": "red", "width": 2},
            hovertemplate="Epoch %{x}<br>Val Loss: %{y:.3f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{metrics['model']} Training History ({metrics['epochs']} epochs)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        showlegend=True,
        legend={"x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top"},
    )

    return fig


def create_map_comparison(all_metrics: list[dict[str, Any]]) -> go.Figure:
    """Create mAP comparison bar chart across models.

    :param all_metrics: List of metrics dictionaries for all models
    :return: Plotly figure with grouped bar chart comparing mAP scores
    :rtype: go.Figure
    """
    if not all_metrics:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No training metrics available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
        return fig

    models = [m["model"] for m in all_metrics]
    map50 = [m.get("final_map50", 0.0) for m in all_metrics]
    map_full = [m.get("final_map", 0.0) for m in all_metrics]

    fig = go.Figure(
        data=[
            go.Bar(
                name="mAP@0.5",
                x=models,
                y=map50,
                text=[f"{v:.1%}" for v in map50],
                textposition="auto",
                marker_color="rgb(55, 83, 109)",
                hovertemplate="mAP@0.5: %{y:.3f}<extra></extra>",
            ),
            go.Bar(
                name="mAP@0.5:0.95",
                x=models,
                y=map_full,
                text=[f"{v:.1%}" for v in map_full],
                textposition="auto",
                marker_color="rgb(26, 118, 255)",
                hovertemplate="mAP@0.5:0.95: %{y:.3f}<extra></extra>",
            ),
        ]
    )

    fig.update_layout(
        title="Model Performance Comparison (mAP)",
        xaxis_title="Model",
        yaxis_title="mAP Score",
        yaxis={"range": [0, 1.0], "tickformat": ".0%"},
        barmode="group",
        template="plotly_white",
        height=400,
        showlegend=True,
    )

    return fig


def create_inference_benchmark_chart(
    benchmark_results: dict[str, dict[str, float]],
) -> go.Figure:
    """Create inference benchmark comparison chart.

    :param benchmark_results: Dictionary mapping model names to benchmark statistics
    :return: Plotly figure with inference time comparison
    :rtype: go.Figure
    """
    models = list(benchmark_results.keys())
    mean_times = [stats["mean"] * 1000 for stats in benchmark_results.values()]
    std_times = [stats["std"] * 1000 for stats in benchmark_results.values()]

    fig = go.Figure(
        data=[
            go.Bar(
                x=models,
                y=mean_times,
                error_y={"type": "data", "array": std_times},
                text=[f"{v:.1f}ms" for v in mean_times],
                textposition="auto",
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(models)],
                hovertemplate="Mean: %{y:.2f}ms<br>Std: %{error_y.array:.2f}ms<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Inference Speed Benchmark (100 runs)",
        xaxis_title="Model",
        yaxis_title="Inference Time (ms)",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    return fig


def create_metrics_summary_table(all_metrics: list[dict[str, Any]]) -> go.Figure:
    """Create summary table of all model metrics.

    :param all_metrics: List of metrics dictionaries for all models
    :return: Plotly table figure with metrics summary
    :rtype: go.Figure
    """
    if not all_metrics:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Prepare table data
    models = [m["model"] for m in all_metrics]
    epochs = [m.get("epochs", 0) for m in all_metrics]
    map50 = [f"{m.get('final_map50', 0.0):.3f}" for m in all_metrics]
    map_full = [f"{m.get('final_map', 0.0):.3f}" for m in all_metrics]
    final_train_loss = [
        f"{m['train_loss'][-1]:.3f}" if m.get("train_loss") else "N/A"
        for m in all_metrics
    ]
    final_val_loss = [
        f"{m['val_loss'][-1]:.3f}" if m.get("val_loss") else "N/A" for m in all_metrics
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": [
                        "Model",
                        "Epochs",
                        "mAP@0.5",
                        "mAP@0.5:0.95",
                        "Final Train Loss",
                        "Final Val Loss",
                    ],
                    "fill_color": "paleturquoise",
                    "align": "left",
                    "font": {"size": 12, "color": "black"},
                },
                cells={
                    "values": [
                        models,
                        epochs,
                        map50,
                        map_full,
                        final_train_loss,
                        final_val_loss,
                    ],
                    "fill_color": "lavender",
                    "align": "left",
                    "font": {"size": 11},
                },
            )
        ]
    )

    fig.update_layout(
        title="Training Metrics Summary",
        height=200 + (len(all_metrics) * 30),
    )

    return fig
