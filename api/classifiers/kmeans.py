import csv
from pathlib import Path

import numpy as np

class KMeans: 
    # DETEMRINISTIC KMEANS
    NUMERIC_FEATURES = (
        "hp", "attack", "defense", "sp_attack", "sp_defense", "speed", 
        "height_m", "weight_kg"
        )

    EXCLUDED_COLUMNS = {"pokedex_id", "name", "type_1"}

    def __init__(self, k = 5, max_iterations: int = 100):
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be a positive integer >= 2")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer >= 1")
        self.k = k
        self.max_iterations = max_iterations
        self.dataset_path = Path(__file__).resolve().parent.parent / "data" / "pokemon_onehot_keep_type.csv"
        self.names = []
        self.types = []
        self.feature_names = []
        self.features = None
        self.centroids = None
        self.assignments = None
        self.clusters = []
        self.iterations = 0
        self.converged = False
        self.silhouette_score = None
        self.history = []

    def load_dataset(self):
        # Load preprocessed CSV and standardize continuous numeric columns.
        with self.dataset_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV does not have a header")
            
            header = reader.fieldnames
            excluded = self.EXCLUDED_COLUMNS
            
            self.feature_names = [
                name for name in header
                if name not in excluded and name != "type_1"
            ]
            
            if not self.feature_names:
                raise ValueError("No numeric features found")
            records = list(reader)

        if self.k > len(records):
            raise ValueError(f"k={self.k} exceeds the number of Pokémon ({len(records)})")
        self.names = [row["name"] for row in records]
        self.types = [row.get("type_1", "") for row in records]
        
        try:
            features = np.array(
                [[float(row[col]) for col in self.feature_names] for row in records],
                dtype=float,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Dataset has an invalid numeric feature: {exc}") from exc
        
        if not np.isfinite(features).all():
            raise ValueError("Dataset contains NaN or infinite feature values")
        
        for col in self.NUMERIC_FEATURES:
            if col in self.feature_names:
                j = self.feature_names.index(col)
                mean = features[:, j].mean()
                std = features[:, j].std()
                features[:, j] = (features[:, j] - mean) / std if std else 0.0
        
        self.features = features
        
        return features

    @staticmethod
    def euclidean_distance(document, centroid):
        # Distance between two Pokémon feature vectors.
        return float(np.linalg.norm(np.asarray(document) - np.asarray(centroid)))

    def fit(self):
        # Fit the K-means model to the dataset.
        if self.features is None:
            self.load_dataset()

        points = self.features
        self.centroids = points[:self.k].copy()
        previous = None
        self.history = []
        self.converged = False

        for iteration in range(1, self.max_iterations + 1):
            # squared distance gives same nearest centroid as Euclidean distance
            squared_distances = np.sum((points[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
            assignments = np.argmin(squared_distances, axis=1)
            
            self.iterations = iteration
            
            self.history.append({
                "iteration": iteration,
                "cluster_sizes": [int(np.count_nonzero(assignments == j)) for j in range(self.k)],
            })

            self.assignments = assignments
            if previous is not None and np.array_equal(assignments, previous):
                self.converged = True
                break

            for j in range(self.k):
                members = points[assignments == j]

                if len(members): 
                    self.centroids[j] = members.mean(axis=0)

            previous = assignments.copy()

        self.clusters = [np.flatnonzero(self.assignments == j).tolist() for j in range(self.k)]
        self.silhouette_score = self.calculate_silhouette()

        return self

    def calculate_silhouette(self):
        # Calculate the silhouette score for the clustering.
        if self.assignments is None or self.centroids is None:
            raise ValueError("Model must be fitted before calculating silhouette score")

        X = self.features
        squared_norms = np.sum(X ** 2, axis=1)
        squared = np.maximum(
            squared_norms[:, None] + squared_norms[None, :] - 2 * np.dot(X, X.T), 0.0
        )

        distances = np.sqrt(squared)

        np.fill_diagonal(distances, 0.0)

        scores = np.zeros(len(X), dtype=float)

        nonempty = [group for group in self.clusters if group]

        if len(nonempty) < 2:
            return 0.0

        for cluster in nonempty:
            if len(cluster) <= 1:
                continue

            rows = np.array(cluster)

            a = distances[np.ix_(rows, rows)].sum(axis=1) / (len(rows) - 1)

            b = np.min(
                np.stack([
                    distances[np.ix_(rows, other)].mean(axis=1)
                    for other in nonempty if other is not cluster
                ], axis=1), axis=1,
            )

            denom = np.maximum(a, b)

            scores[rows] = np.divide(b - a, denom, out=np.zeros_like(a), where=denom > 0)

        return float(scores.mean())

    def summary(self, top_n=5):
        # Return a summary of the clustering results, including top features and Pokémon in each cluster.
        if self.assignments is None:
            raise RuntimeError("Run fit() first")
        
        groups = []

        for j, indices in enumerate(self.clusters):
            top_indices = np.argsort(-self.centroids[j], kind="stable")[:top_n]

            groups.append({
                "cluster": j + 1,
                "count": len(indices),
                "top_features": [self.feature_names[i] for i in top_indices],
                "pokemon": [self.names[i] for i in indices],
            })

        return {
            "total_pokemon": len(self.names),
            "total_features": len(self.feature_names),
            "k": self.k,
            "iterations": self.iterations,
            "converged": self.converged,
            "iteration_history": self.history,
            "overall_silhouette_score": self.silhouette_score,
            "clusters": groups,
        }

    def save_assignments(self, output_path=None):
        # Save the cluster assignments to a CSV file.
        if self.assignments is None:
            raise RuntimeError("Run fit() first")
        
        output = Path(output_path) if output_path else (
            Path(__file__).resolve().parent.parent / "data" / "kmeans_clusters.csv"
        )

        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([self.silhouette_score])
            writer.writerow(["name", "type_1", "cluster"])
            for name, kind, cluster in zip(self.names, self.types, self.assignments):
                writer.writerow([name, kind, int(cluster) + 1])

        return output