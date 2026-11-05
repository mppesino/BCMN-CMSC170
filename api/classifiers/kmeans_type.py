import csv
from collections import Counter
from pathlib import Path

import numpy as np


class KMeans:
    # Deterministic K-means used as a cluster-then-label classifier for type_1.
    NUMERIC_FEATURES = (
        "hp", "attack", "defense", "sp_attack", "sp_defense", "speed",
        "height_m", "weight_kg",
        "is_hidden_ability", "ability_1"
    )

    EXCLUDED_COLUMNS = {"pokedex_id", "name", "type_1"}

    def __init__(self, k: int = 18, max_iterations: int = 100, seed: int = 42):
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer >= 1")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer >= 1")
        self.k = k
        self.max_iterations = max_iterations
        self.seed = seed
        self.dataset_path = Path(__file__).resolve().parent.parent / "data" / "pokemon_onehot_keep_type.csv"
        self.names = []
        self.types = []
        self.feature_names = []
        self.raw_features = None
        self.features = None          # standardized with TRAIN statistics
        self.centroids = None
        self.assignments = None       # cluster id for every Pokemon (train + test)
        self.train_idx = None
        self.clusters = []            # global indices of TRAIN members per cluster
        self.cluster_types = []
        self.cluster_type_counts = []
        self.iterations = 0
        self.converged = False
        self.silhouette_score = None  # computed on train points only
        self.history = []

    # ------------------------------------------------------------------ data
    def load_dataset(self):
        # Load the preprocessed CSV. Standardization happens in fit() using train stats.
        with self.dataset_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV does not have a header")

            self.feature_names = [
                name for name in reader.fieldnames
                if name not in self.EXCLUDED_COLUMNS
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

        self.raw_features = features
        self.features = features.copy()
        return features

    def _standardize(self, train_idx):
        # z-score continuous columns using TRAIN mean/std only (no test leakage).
        features = self.raw_features.copy()
        for col in self.NUMERIC_FEATURES:
            if col in self.feature_names:
                j = self.feature_names.index(col)
                mean = self.raw_features[train_idx, j].mean()
                std = self.raw_features[train_idx, j].std()
                features[:, j] = (self.raw_features[:, j] - mean) / std if std else 0.0
        self.features = features

    @staticmethod
    def sequential_split(n, train_ratio=0.70):
        # Same split as the KNN classifier: first 70% of rows = train, remaining 30% = test.
        n_train = int(n * train_ratio)
        return np.arange(n_train, dtype=int), np.arange(n_train, n, dtype=int)

    # ------------------------------------------------------------------- fit
    def assign_clusters(self, X):
        # Nearest-centroid index for each row of X.
        squared = np.sum((X[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        return np.argmin(squared, axis=1)

    def fit(self, train_idx=None):
        # Fit on train_idx only (all rows if None). Labels are never used for clustering.
        if self.raw_features is None:
            self.load_dataset()

        n = len(self.names)
        train_idx = np.arange(n) if train_idx is None else np.asarray(train_idx, dtype=int)
        if self.k > len(train_idx):
            raise ValueError(f"k={self.k} exceeds the number of training Pokémon ({len(train_idx)})")

        self.train_idx = train_idx
        self._standardize(train_idx)
        points = self.features[train_idx]

        rng = np.random.default_rng(self.seed)
        self.centroids = points[rng.choice(len(points), self.k, replace=False)].copy()
        previous = None
        self.history = []
        self.converged = False
        local = None

        for iteration in range(1, self.max_iterations + 1):
            local = self.assign_clusters(points)
            self.iterations = iteration
            self.history.append({
                "iteration": iteration,
                "cluster_sizes": [int(np.count_nonzero(local == j)) for j in range(self.k)],
            })

            if previous is not None and np.array_equal(local, previous):
                self.converged = True
                break

            for j in range(self.k):
                members = points[local == j]
                if len(members):
                    self.centroids[j] = members.mean(axis=0)

            previous = local.copy()

        # Train-only clusters (global indices), plus assignments for every Pokemon.
        self.clusters = [train_idx[np.flatnonzero(local == j)].tolist() for j in range(self.k)]
        self.assignments = self.assign_clusters(self.features)

        self.assign_cluster_types()
        self.silhouette_score = self.calculate_silhouette(points, local)
        return self

    def assign_cluster_types(self):
        # Label each cluster with the most frequent TRAIN type_1 (ties: alphabetical).
        # Empty clusters fall back to the overall train majority type.
        overall = Counter(self.types[i] for i in self.train_idx)
        fallback = sorted(overall.items(), key=lambda item: (-item[1], item[0]))[0][0]

        self.cluster_types = []
        self.cluster_type_counts = []
        for cluster in self.clusters:
            counts = Counter(self.types[i] for i in cluster if self.types[i])
            if not counts:
                self.cluster_types.append(fallback)
                self.cluster_type_counts.append({})
                continue
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            self.cluster_types.append(ordered[0][0])
            self.cluster_type_counts.append(dict(ordered))
        return self.cluster_types

    # --------------------------------------------------------------- predict
    def predict(self, indices):
        # Predicted type_1 for the given Pokemon indices.
        if self.centroids is None:
            raise RuntimeError("Run fit() first")
        indices = np.asarray(indices, dtype=int)
        cluster_ids = self.assign_clusters(self.features[indices])
        return [self.cluster_types[c] for c in cluster_ids]

    @staticmethod
    def accuracy(y_true, y_pred):
        if len(y_true) == 0:
            return 0.0
        return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    @staticmethod
    def macro_f1(y_true, y_pred):
        # Unweighted mean of per-class F1 over every class seen in y_true or y_pred.
        labels = sorted(set(y_true) | set(y_pred))
        f1s = []
        for label in labels:
            tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
            fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
            fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return float(np.mean(f1s))

    def evaluate(self, indices):
        y_true = [self.types[i] for i in indices]
        y_pred = self.predict(indices)
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        return {
            "accuracy": self.accuracy(y_true, y_pred),
            "macro_f1": self.macro_f1(y_true, y_pred),
            "correct": correct,
            "incorrect": len(y_true) - correct,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def print_incorrect(self, indices, result):
        # Same line format as the KNN classifier.
        for i, actual, predicted in zip(indices, result["y_true"], result["y_pred"]):
            if actual != predicted:
                print("Pokémon:", self.names[i], "| Actual:", actual, "| Predicted:", predicted)

    def print_cluster_assignments(self):
        # For every cluster: assigned type, train purity, and its train/test members.
        train_set = set(self.train_idx.tolist())
        print("\nCluster Assignments:")

        for j in range(self.k):
            members = np.flatnonzero(self.assignments == j).tolist()
            train_members = [self.names[i] for i in members if i in train_set]
            test_members = [self.names[i] for i in members if i not in train_set]

            label = self.cluster_types[j]
            top_count = self.cluster_type_counts[j].get(label, 0)

            print(f"\nCluster {j + 1} | Assigned Type: {label} "
                  f"({top_count}/{len(train_members)} training Pokémon)")
            print(f"  Training ({len(train_members)}):", ", ".join(train_members) if train_members else "none")
            print(f"  Testing  ({len(test_members)}):", ", ".join(test_members) if test_members else "none")

    # ---------------------------------------------------------------- scores
    def calculate_silhouette(self, X, labels):
        # Silhouette on the given points (local labels).
        if self.centroids is None:
            raise ValueError("Model must be fitted before calculating silhouette score")

        squared_norms = np.sum(X ** 2, axis=1)
        squared = np.maximum(
            squared_norms[:, None] + squared_norms[None, :] - 2 * np.dot(X, X.T), 0.0
        )
        distances = np.sqrt(squared)
        np.fill_diagonal(distances, 0.0)

        scores = np.zeros(len(X), dtype=float)
        nonempty = [np.flatnonzero(labels == j) for j in range(self.k)]
        nonempty = [group for group in nonempty if len(group)]

        if len(nonempty) < 2:
            return 0.0

        for cluster in nonempty:
            if len(cluster) <= 1:
                continue

            a = distances[np.ix_(cluster, cluster)].sum(axis=1) / (len(cluster) - 1)
            b = np.min(
                np.stack([
                    distances[np.ix_(cluster, other)].mean(axis=1)
                    for other in nonempty if other is not cluster
                ], axis=1), axis=1,
            )
            denom = np.maximum(a, b)
            scores[cluster] = np.divide(b - a, denom, out=np.zeros_like(a), where=denom > 0)

        return float(scores.mean())

    def summary(self, top_n=5):
        if self.assignments is None:
            raise RuntimeError("Run fit() first")

        groups = []
        for j, indices in enumerate(self.clusters):
            top_indices = np.argsort(-self.centroids[j], kind="stable")[:top_n]
            groups.append({
                "cluster": j + 1,
                "count": len(indices),
                "assigned_type": self.cluster_types[j],
                "type_counts": self.cluster_type_counts[j],
                "top_features": [self.feature_names[i] for i in top_indices],
                "pokemon": [self.names[i] for i in indices],
            })

        return {
            "total_pokemon": len(self.names),
            "train_pokemon": len(self.train_idx),
            "total_features": len(self.feature_names),
            "k": self.k,
            "iterations": self.iterations,
            "converged": self.converged,
            "iteration_history": self.history,
            "train_silhouette_score": self.silhouette_score,
            "clusters": groups,
        }

    def save_assignments(self, output_path=None):
        if self.assignments is None:
            raise RuntimeError("Run fit() first")

        output = Path(output_path) if output_path else (
            Path(__file__).resolve().parent.parent / "data" / "kmeans_clusters.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        train_set = set(self.train_idx.tolist())

        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "type_1", "cluster", "cluster_type", "split"])
            for i, (name, kind, cluster) in enumerate(zip(self.names, self.types, self.assignments)):
                writer.writerow([
                    name, kind, int(cluster) + 1, self.cluster_types[int(cluster)],
                    "train" if i in train_set else "test",
                ])
        return output


def select_k(base, train_idx, candidates=(18, 36, 54, 72, 90, 108)):
    # Choose k by validation macro-F1 on a slice of TRAIN (test stays untouched).
    inner_n = int(len(train_idx) * 0.80)
    inner_train, val_idx = train_idx[:inner_n], train_idx[inner_n:]

    best_k, best_f1 = None, -1.0
    for k in candidates:
        if k > len(inner_train):
            continue
        m = KMeans(k=k).fit(inner_train)
        score = m.evaluate(val_idx)["macro_f1"]
        if score > best_f1:
            best_k, best_f1 = k, score
    return best_k


def main():
    base = KMeans()
    base.load_dataset()

    # Same split as the KNN classifier: first 70% train, last 30% test.
    train_idx, test_idx = KMeans.sequential_split(len(base.names), train_ratio=0.70)

    print("\nData set loaded successfully.")
    print("Total numbers of records:", len(base.names))
    print("Training instances:", len(train_idx))
    print("Testing instances:", len(test_idx))

    best_k = select_k(base, train_idx)

    # Final model: refit on the full train set, evaluate on test once.
    model = KMeans(k=best_k).fit(train_idx)
    test_result = model.evaluate(test_idx)

    print("\nK-Means Classification Results")
    print("k =", model.k)
    print("\nCorrect Predictions:", test_result["correct"])
    print("Incorrect Predictions:", test_result["incorrect"])
    print("Classification Accuracy:", round(test_result["accuracy"] * 100, 2), "%")
    print("Macro F1:", round(test_result["macro_f1"], 4))

    print("\nIncorrect Classifications:")
    model.print_incorrect(test_idx, test_result)

    model.print_cluster_assignments()

    output = model.save_assignments()
    print("\nSaved cluster assignments to:", output)


if __name__ == "__main__":
    main()