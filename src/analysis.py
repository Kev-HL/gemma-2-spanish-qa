"""
Visualization (charts, results, etc) and other helper functions for the analysis of
results from the training and evaluation runs.
"""

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import wandb


def plot_wandb_runs(
    wandb_api: wandb.Api,
    entity: str,
    project: str,
    comparison_name: str,
    run_ids: list[str],
    aliases: dict[str, str] = {},
    metrics: list[str] = ["train/loss", "eval/loss", "eval/f1", "eval/exact_match"],
) -> None:
    """
    Plot W&B run metrics for comparison.

    Args:
        wandb_api: Initialized wandb.Api() instance
        entity: W&B entity (username)
        project: W&B project name
        comparison_name: Title/name for the comparison
        run_ids: List of run IDs to plot (or single run)
        metrics: List of metrics to plot (or single metric)
        aliases: Optional dict with key-value pair of run name and alias

    Returns:
        None (displays plots)
    """
    # Safety check
    if not isinstance(run_ids, list):
        if isinstance(run_ids, str):
            # Safety check for single run without list, create list of 1
            run_ids = [run_ids]
        else:
            raise ValueError("run_ids should be a single string or list of strings.")

    if not isinstance(metrics, list):
        if isinstance(metrics, str):
            # Safety check for single metric without list, create list of 1
            metrics = [metrics]
        else:
            raise ValueError("metrics should be a single string or list of strings.")

    # Set dark style
    plt.style.use("dark_background")

    # Set cycle colors
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Set1.colors)

    # Loop for each metric and create a plot for it
    for metric in metrics:

        # Initialize plot
        fig, ax = plt.subplots(figsize=(8, 4))

        # Loop for each run and add it to the metric's plot
        for run_id in run_ids:
            # Get run values
            run = wandb_api.run(f"{entity}/{project}/{run_id}")
            history = run.history()
            plot_values = history[["_step", metric]].dropna()
            marker_size = 4 if len(plot_values) < 30 else 1

            # Use alias if available if not run name
            label = aliases[run.name] if run.name in aliases.keys() else run.name

            # Plot run
            ax.plot(
                plot_values["_step"],
                plot_values[metric],
                linewidth=1.5,
                marker="o",
                markersize=marker_size,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=label,
            )

        # Grid
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.7, color="white")
        ax.set_axisbelow(True)

        # Labels and title
        ax.set_xlabel("Training Step", fontsize=12, fontweight="bold", color="white")
        ax.set_ylabel(f"{metric} Values", fontsize=12, fontweight="bold", color="white")
        ax.set_title(
            f"{metric} - {comparison_name}",
            fontsize=14,
            fontweight="bold",
            color="white",
            pad=20,
        )

        # Spine styling
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(1)

        # Background
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#0f0f0f")

        # Tick styling
        ax.tick_params(colors="white", labelsize=10)

        # Legend
        ax.legend(
            loc="best",
            framealpha=0.9,
            facecolor="#1a1a1a",
            edgecolor="white",
            fontsize=10,
        )

        # Display plot (inside loop, display each individually to avoid Jupyter issues)
        plt.tight_layout()
        plt.show()


def get_wandb_best_values(
    wandb_api: wandb.Api,
    entity: str,
    project: str,
    run_ids: list[str],
    aliases: dict[str, str] = {},
    metrics: list[str] = ["eval/loss", "eval/f1", "eval/exact_match"],
    best_metric: str = "eval/f1",
    best_order: str = "max",
) -> pd.DataFrame:
    """
    Get best values from run according to a best metric

    Args:
        wandb_api: Initialized wandb.Api() instance
        entity: W&B entity (username)
        project: W&B project name
        run_ids: List of run IDs to retrieve values from
        aliases: Optional dict with key-value pair of run name and alias
        metrics: List of metrics to extract the values of
        best_metric: Key to be used to determine 'best' entry
        best_order: 'max' for descending order (more is better), 'min' for ascending

    Returns:
        Pandas DataFrame with best values per run, sorted by best metric
    """
    RUN_NAME_KEY = "run_name"

    if not isinstance(run_ids, list):
        if isinstance(run_ids, str):
            # Safety check for single run without list, create list of 1
            run_ids = [run_ids]
        else:
            raise ValueError("run_ids should be a single string or list of strings.")

    if not isinstance(metrics, list):
        if isinstance(metrics, str):
            # Safety check for single metric without list, create list of 1
            metrics = [metrics]
        else:
            raise ValueError("metrics should be a single string or list of strings.")

    if best_order not in ["min", "max"]:
        raise ValueError(f'Expecting best_order "max" or "min", got {best_order}')

    ascending = False if best_order == "max" else True
    records = []
    for run_id in run_ids:
        # Evaluation run (no history, just final metrics)
        run = wandb_api.run(f"{entity}/{project}/{run_id}")

        # Get history (all values)
        history = run.history()

        # Sort values
        if best_metric in history.columns:
            history.sort_values(by=best_metric, ascending=ascending, inplace=True)
        else:
            raise ValueError(
                f"Best metric {best_metric} does not exist in run {run.name}"
            )

        # Filter un-requested columns
        if set(metrics).issubset(history.columns):
            history = history[metrics]
        else:
            raise ValueError(
                f"One or more of {metrics} is not present on run {run.name}"
            )

        # Extract best values
        best_values = history.iloc[0].to_dict()
        best_values[RUN_NAME_KEY] = (
            aliases[run.name] if run.name in aliases.keys() else run.name
        )

        # Append to records
        records.append(best_values)

    # Convert records to Pandas DataFrame and sort values
    records_df = pd.DataFrame.from_records(records)
    if best_metric in records_df.columns:
        records_df.sort_values(by=best_metric, ascending=ascending, inplace=True)
    records_df = records_df[[RUN_NAME_KEY] + metrics]

    # Return Pandas DataFrame
    return records_df.reset_index(drop=True)
