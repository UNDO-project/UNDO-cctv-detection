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

    # Check if we have epoch data (DETR) or just sequential indices (YOLO)
    train_epochs = metrics.get("train_epochs")
    val_epochs = metrics.get("val_epochs")

    # Use epoch values if available, otherwise use sequential indices
    train_x = train_epochs or list(range(1, len(metrics["train_loss"]) + 1))
    val_x = val_epochs or list(range(1, len(metrics["val_loss"]) + 1))

    # Add training loss trace
    fig.add_trace(
        go.Scatter(
            x=train_x,
            y=metrics["train_loss"],
            mode="lines",
            name="Training Loss",
            line={"color": "blue", "width": 2},
            hovertemplate="Epoch %{x:.1f}<br>Train Loss: %{y:.3f}<extra></extra>",
        )
    )

    # Add validation loss trace
    fig.add_trace(
        go.Scatter(
            x=val_x,
            y=metrics["val_loss"],
            mode="lines",
            name="Validation Loss",
            line={"color": "red", "width": 2},
            hovertemplate="Epoch %{x:.1f}<br>Val Loss: %{y:.3f}<extra></extra>",
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
    """Create summary table showing best run for each model.

    Filters out duplicate runs and shows only the best performing run
    (highest mAP@0.5) for each model type.

    :param all_metrics: List of metrics dictionaries for all models
    :return: Plotly table figure with best metrics per model
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

    # Group by model and keep only the best run
    # For models with mAP: highest mAP@0.5
    # For models without mAP (DETR): most epochs trained
    best_runs = {}
    for m in all_metrics:
        model_name = m["model"]
        map50_val = m.get("final_map50", 0.0)
        epochs = m.get("epochs", 0)

        if model_name not in best_runs:
            best_runs[model_name] = m
        else:
            # If mAP is available (>0), use it for comparison
            # Otherwise use epochs (more training = better)
            existing_map = best_runs[model_name].get("final_map50", 0.0)
            existing_epochs = best_runs[model_name].get("epochs", 0)

            if map50_val > 0 and existing_map > 0:
                # Both have mAP, compare mAP
                if map50_val > existing_map:
                    best_runs[model_name] = m
            else:
                # No mAP available, compare epochs
                if epochs > existing_epochs:
                    best_runs[model_name] = m

    # Convert back to list and sort
    # Models with mAP first (sorted by mAP), then models without (sorted by epochs)
    filtered_metrics = sorted(
        best_runs.values(),
        key=lambda x: (x.get("final_map50", 0.0), x.get("epochs", 0)),
        reverse=True,
    )

    # Prepare table data
    models = [m["model"] for m in filtered_metrics]
    epochs_list = [m.get("epochs", 0) for m in filtered_metrics]
    map50 = [f"{m.get('final_map50', 0.0):.3f}" for m in filtered_metrics]
    map_full = [f"{m.get('final_map', 0.0):.3f}" for m in filtered_metrics]
    final_train_loss = [
        f"{m['train_loss'][-1]:.3f}" if m.get("train_loss") else "N/A"
        for m in filtered_metrics
    ]
    final_val_loss = [
        f"{m['val_loss'][-1]:.3f}" if m.get("val_loss") else "N/A"
        for m in filtered_metrics
    ]

    # Add status indicators
    status = []
    for m in filtered_metrics:
        map50_val = m.get("final_map50", 0.0)
        map_full_val = m.get("final_map", 0.0)
        m_epochs = m.get("epochs", 0)

        # Check if training seems complete
        if map50_val == 0.0 and map_full_val == 0.0:
            # Models like DETR that don't compute mAP during training
            if m_epochs > 0:
                status.append("📊 mAP Not Computed")
            else:
                status.append("⚠️ No Training")
        elif map50_val >= 0.7:
            status.append("✅ Excellent")
        elif map50_val >= 0.5:
            status.append("✅ Good")
        else:
            status.append("⚠️ Needs Training")

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
                        "Status",
                    ],
                    "fill_color": "paleturquoise",
                    "align": "left",
                    "font": {"size": 12, "color": "black"},
                },
                cells={
                    "values": [
                        models,
                        epochs_list,
                        map50,
                        map_full,
                        final_train_loss,
                        final_val_loss,
                        status,
                    ],
                    "fill_color": "lavender",
                    "align": "left",
                    "font": {"size": 11},
                },
            )
        ]
    )

    fig.update_layout(
        title="Training Metrics Summary (Best Runs Only)",
        height=150 + (len(filtered_metrics) * 30),
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )

    return fig
