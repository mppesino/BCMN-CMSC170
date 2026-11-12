
import numpy as np
import matplotlib.pyplot as plt

from evaluator import Evaluator
from pathlib import Path


def bar_graph(results, save_path=None):
    """
    Displays bar graphs comparing KNN, K-Means,
    and Naive Bayes using four metrics:

    1. Accuracy
    2. Precision
    3. Recall
    4. F1-score
    """

    evaluator = Evaluator()

    algorithms = ["KNN", "K-Means", "Naive Bayes"]

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for algorithm in algorithms:

        confusion_matrix, mini_trix = results[algorithm]

        # Accuracy from the 18x18 confusion matrix.
        matrix = np.asarray(confusion_matrix)

        total = np.sum(matrix)
        correct = np.trace(matrix)

        accuracy = correct / total if total else 0

        # Precision and Recall using Bea's functions.
        precisions = []
        recalls = []

        for entry in mini_trix:
            for pokemon_type, type_matrix in entry.items():

                precision = evaluator.get_precision(type_matrix)
                recall = evaluator.get_recall(type_matrix)

                precisions.append(precision)
                recalls.append(recall)

        mean_precision = np.mean(precisions) if precisions else 0
        mean_recall = np.mean(recalls) if recalls else 0

        # Mean F1 using Bea's function.
        mean_f1, per_type_f1 = evaluator.get_f1_mean(mini_trix)

        accuracy_scores.append(accuracy * 100)
        precision_scores.append(mean_precision * 100)
        recall_scores.append(mean_recall * 100)
        f1_scores.append(mean_f1 * 100)

    metrics = {
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
        "Recall": recall_scores,
        "F1-score": f1_scores
    }

    # Four bar graphs in one figure.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for ax, (metric, scores) in zip(axes.flat, metrics.items()):

        bars = ax.bar(
            algorithms,
            scores,
            color=colors,
            width=0.6
        )

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{score:.2f}%",
                ha="center"
            )

    fig.suptitle(
        "Pokelytics: Classification Algorithm Comparison",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return metrics

def confusion_matrix_colored(
    confusion_matrix,
    algorithm,
    save_path=None
):

    pokemon_types = [
        "Normal", "Fire", "Water", "Electric",
        "Grass", "Ice", "Fighting", "Poison",
        "Ground", "Flying", "Psychic", "Bug",
        "Rock", "Ghost", "Dragon", "Dark",
        "Steel", "Fairy"
    ]

    matrix = np.asarray(confusion_matrix)

    if matrix.shape != (18, 18):
        raise ValueError(
            "Confusion matrix must have an 18x18 shape."
        )

    # Create the figure.
    fig, ax = plt.subplots(figsize=(14, 12))

    # Apply blue color weighting.
    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0
    )

    # Add a color scale.
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Number of Pokemon")

    # Label the axes.
    ax.set_xticks(np.arange(18))
    ax.set_yticks(np.arange(18))

    ax.set_xticklabels(pokemon_types)
    ax.set_yticklabels(pokemon_types)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    ax.set_xlabel("Predicted Type", fontsize=12)
    ax.set_ylabel("Actual Type", fontsize=12)

    ax.set_title(
        f"{algorithm} - Confusion Matrix",
        fontsize=16,
        fontweight="bold",
        pad=20
    )

    # Display the numerical count in each cell.
    threshold = matrix.max() / 2

    for i in range(18):
        for j in range(18):

            value = int(matrix[i, j])

            text_color = (
                "white" if value > threshold else "black"
            )

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8
            )

    fig.tight_layout()

    # Save the visualization if a path is provided.
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Confusion matrix saved to: {save_path}")

    plt.show()
    plt.close(fig)

    return fig
