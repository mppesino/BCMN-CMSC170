
import numpy as np
import matplotlib.pyplot as plt

from evaluator import Evaluator


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

                precision = evaluator.precision(type_matrix)
                recall = evaluator.recall(type_matrix)

                precisions.append(precision)
                recalls.append(recall)

        mean_precision = np.mean(precisions) if precisions else 0
        mean_recall = np.mean(recalls) if recalls else 0

        # Mean F1 using Bea's function.
        mean_f1, per_type_f1 = evaluator.f1_mean(mini_trix)

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
