
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

from knn import KNNclassifier
from kmeans import KMeans
from nb import NaiveBayes
from api.services.outputs.visualization import bar_graph, confusion_matrix_colored

# Get paths relative to this script.
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET = BASE_DIR / "data" / "pokemon_onehot_keep_type.csv"
OUTPUT = BASE_DIR / "outputs" / "model_comparison.png"


def main():
    results = {}

    # 1. Run KNN
    print("\nRunning KNN...")

    knn = KNNclassifier(0, 0)

    data = knn.load_data(DATASET)
    knn.build_training_and_test_sets(data)

    knn_results = knn.run_knn()

    results["KNN"] = knn_results

    print("\nKNN completed.")

    # 2. Run K-Means
    print("Running K-Means...")

    kmeans = KMeans()

    with redirect_stdout(StringIO()):
        kmeans_results = kmeans.run_classifier()

    results["K-Means"] = kmeans_results

    print("K-Means completed.")

    # 3. Run Naive Bayes
    print("Running Naive Bayes...")

    nb = NaiveBayes(smooth=1.0)

    with redirect_stdout(StringIO()):
        nb_results = nb.naive_bayes()

    results["Naive Bayes"] = nb_results

    print("Naive Bayes completed.")

    # 4. Generate bar graphs
    print("Generating comparison graphs...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    bar_graph(
        results,
        save_path=OUTPUT
    )

    print(f"Graph saved to: {OUTPUT}")

    bar_graph(
        results,
        save_path=OUTPUT
    )

    print(f"Graph saved to: {OUTPUT}")

    # 5. Generate colored confusion matrices
    print("\nGenerating colored confusion matrices...")

    for algorithm, classifier_results in results.items():

        # (18x18 confusion matrix, 2x2 matrices per type)
        confusion_matrix, mini_trix = classifier_results

        filename = (
            algorithm.lower()
            .replace("-", "")
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        output_path = BASE_DIR / "outputs" / filename

        confusion_matrix_colored(
            confusion_matrix,
            algorithm=algorithm,
            save_path=output_path
        )

    print("\nAll visualizations generated successfully.")

if __name__ == "__main__":
    main()
