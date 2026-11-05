
import argparse
from classifiers.kmeans import KMeans


def main():
    parser = argparse.ArgumentParser(
        description="Run K-Means clustering on Pokemon data"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of clusters"
    )

    args = parser.parse_args()

    print("Starting K-Means clustering...")
    print(f"Number of clusters: {args.k}")

    model = KMeans(k=args.k)

    model.fit()

    print("\nClustering Results:")
    print(model.summary())

    print("\nSaving cluster assignments...")
    model.save_assignments()

    print("\nK-Means completed successfully!")


if __name__ == "__main__":
    main()
