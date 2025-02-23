import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_FOLDER = "results"


def plot_metric_heatmap_1(file_name: str, title: str):
    df = pd.read_csv(file_name, index_col=0)  # Read the CSV file
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".3f")
    # Explicitly set the title
    if title is not None: plt.suptitle(title, fontsize=14, fontweight='bold')
    # Adjust layout to prevent title clipping
    plt.tight_layout()
    # Show the heatmap
    plt.show()


def plot_time_heatmap_1(file_name: str, title: str, seconds: bool = True):
    df_time = pd.read_csv(file_name, index_col=0)  # Read the CSV file
    # Set up the figure size (larger for better visibility)
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        df_time,
        annot=True,
        cmap="coolwarm",
        fmt=".0f" if seconds else ".3f",  # **Format seconds with 2 decimal places**
        linewidths=1,  # Thicker grid lines
        annot_kws={"size": 12},  # **Smaller font size**
        square=True,  # Ensures all cells are square
        cbar_kws={"shrink": 0.8}  # Shrinks color bar to avoid clutter
    )
    plt.subplots_adjust(top=0.92)
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    if title is not None: plt.suptitle(title, fontsize=14, fontweight='bold')
    # Show the heatmap
    plt.show()

def plot_metric_heatmap_2(file_name: str, title: str):
    df = pd.read_csv(file_name, index_col=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".3f")
    if title is not None: plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_metric_heatmap_1(f"{RESULTS_FOLDER}/smt_metric_1.csv", "SMT Approximate Equivalence Metric - Computed Values")
    plot_time_heatmap_1(f"{RESULTS_FOLDER}/smt_time_1.csv", "SMT Approximate Equivalence Metric - Computational Times")
    plot_metric_heatmap_2(f"{RESULTS_FOLDER}/smt_metric_2.csv", "SMT Approximate Equivalence Metric - Computed Values")
    plot_metric_heatmap_1(f"{RESULTS_FOLDER}/bis_max_metric_1.csv", "Approximate Bisimulation Metric (Max) - Computed Values")
    plot_time_heatmap_1(f"{RESULTS_FOLDER}/bis_max_time_1.csv", "Approximate Bisimulation Metric (Max) - Computational Times", seconds=False)
    plot_metric_heatmap_1(f"{RESULTS_FOLDER}/bis_avg_metric_1.csv","Approximate Bisimulation Metric (Avg) - Computed Values")
    plot_time_heatmap_1(f"{RESULTS_FOLDER}/bis_avg_time_1.csv","Approximate Bisimulation Metric (Avg) - Computational Times", seconds=False)
    plot_metric_heatmap_2(f"{RESULTS_FOLDER}/bis_max_metric_2.csv","Approximate Bisimulation Metric (Max) - Computed Values")
    plot_metric_heatmap_2(f"{RESULTS_FOLDER}/bis_avg_metric_2.csv","Approximate Bisimulation Metric (Avg) - Computed Values")