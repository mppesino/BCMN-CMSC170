from collections import Counter

from evaluator import Evaluator

import numpy as np

NUMERIC_FEATURES = (
    "hp", "attack", "defense", "sp_attack", "sp_defense", "speed",
    "height_m", "weight_kg",
    "is_hidden_ability", "ability_1"
    )

EXCLUDED_COLUMNS = {"pokedex_id", "name", "type_1"}

class KMeans:

    def __init__(self, k:int=18*3, max_iterations:int=100):

        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer >= 1")

        if not isinstance(k, int) or k <= 1:
            raise ValueError("clusters must be a positive integer > 1")

        self.k = k
        self.max_iterations = max_iterations
        self.names = []
        self.types = []
        self.feature_names = []
        self.features = None
        self.centroids = None
        self.assignments = None
        self.clusters = []
        self.cluster_types = []
        self.top_entries = []
        self.iterations = 0
        self.converged = False
        self.silhouette_score = None
        self.correct_count = 0
        self.incorrect = []

        self.test_features = None
        self.test_ids = []
        self.test_names = []
        self.test_types = []

        self.len_data = 0
        self.len_train_data = 0
        self.len_test_data = 0

    def generate_train_test(self, train: list, test: list, lookup: dict):
        self.feature_names = [name for name in lookup if name not in EXCLUDED_COLUMNS]
        feature_idx = [lookup[name] for name in self.feature_names]

        if not self.feature_names:
            raise ValueError("No numeric features found")
        if self.k > len(train):
            raise ValueError(f"k={self.k} exceeds the number of training Pokémon ({len(train)})")
        if self.k < 18:
            raise ValueError(f"k={self.k} must be greater than or equal to number of Pokemon types (18)")

        type_idx = lookup["type_1"]
        name_idx = lookup["name"]
        id_idx = lookup["pokedex_id"]

        try:
            train_features = np.array([[float(row[i]) for i in feature_idx] for row in train], dtype=float)
            test_features = np.array([[float(row[i]) for i in feature_idx] for row in test], dtype=float)
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"Dataset has an invalid numeric feature: {exc}") from exc

        if not (np.isfinite(train_features).all() and np.isfinite(test_features).all()):
            raise ValueError("Dataset contains NaN or infinite feature values")

        # standardize using training statistics only
        for col in NUMERIC_FEATURES:
            if col in self.feature_names:
                j = self.feature_names.index(col)
                mean = train_features[:, j].mean()
                std = train_features[:, j].std()
                train_features[:, j] = (train_features[:, j] - mean) / std if std else 0.0
                test_features[:, j] = (test_features[:, j] - mean) / std if std else 0.0

        self.features = train_features
        self.names = [row[name_idx] for row in train]
        self.types = [row[type_idx] for row in train]

        self.test_features = test_features
        self.test_ids = [row[id_idx] for row in test]
        self.test_names = [row[name_idx] for row in test]
        self.test_types = [row[type_idx] for row in test]

        self.len_data = len(train) + len(test)
        self.len_train_data = len(train)
        self.len_test_data = len(test)

    def label_clusters(self):
        # Assign each cluster its most common type (ties go to the type seen first).
        labels = []
        for cluster in self.clusters:
            counts = Counter(self.types[i] for i in cluster)
            labels.append(counts.most_common(1)[0][0] if counts else "")
        return labels

    def fit(self):
        # Fit the K-means model to the training data.
        points = self.features
        self.centroids = points[:self.k].copy()
        previous = None
        self.converged = False

        for iteration in range(1, self.max_iterations + 1):
            # squared distance gives same nearest centroid as Euclidean distance
            squared_distances = np.sum((points[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
            assignments = np.argmin(squared_distances, axis=1)

            self.iterations = iteration
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
        self.cluster_types = self.label_clusters()

    def evaluate(self):
        # Assign each test Pokémon to its nearest centroid and predict that cluster's type
        sq = np.sum((self.test_features[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        predicted_clusters = np.argmin(sq, axis=1)

        self.correct_count = 0
        self.incorrect = []

        for id, name, actual, c in zip(self.test_ids, self.test_names, self.test_types, predicted_clusters):
            predicted = self.cluster_types[c]
            if predicted == actual:
                self.correct_count += 1
            else:
                self.incorrect.append({
                    "id": id,
                    "pokemon": name,
                    "actual": actual,
                    "predicted": predicted,
                })

    def evaluate_clusters(self, top_n=5):
        # Evaluate the clusters: silhouette score, plus a summary and the
        # top_n Pokémon closest to the centroid of each cluster.
        if self.assignments is None or self.centroids is None:
            raise ValueError("Model must be fitted before evaluating clusters")

        # --- silhouette score ---
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
            self.silhouette_score = 0.0
        else:
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

            self.silhouette_score = float(scores.mean())

        # --- per-cluster summary and top entries ---
        self.top_entries = []

        for j, cluster in enumerate(self.clusters):
            dists = np.linalg.norm(X[cluster] - self.centroids[j], axis=1) if cluster else np.array([])
            order = np.argsort(dists, kind="stable")[:top_n]

            top = [{
                "name": self.names[cluster[o]],
                "type": self.types[cluster[o]],
                "cluster": j + 1,
                "distance": float(dists[o]),
            } for o in order]
            self.top_entries.append(top)

            top_features = [self.feature_names[i] for i in np.argsort(-self.centroids[j], kind="stable")[:top_n]]

            print(f"\nCluster {j + 1} -> most common type: {self.cluster_types[j]} ({len(cluster)} Pokémon)")
            print(f"  Top features: {', '.join(top_features)}")
            for entry in top:
                print(f"  {entry['name']} ({entry['type']}) - distance {entry['distance']:.4f}")

        print(f"\nIterations: {self.iterations} (converged: {self.converged})")
        print(f"Silhouette Score (train): {self.silhouette_score:.4f}")

    def build(self, train: list, test: list, lookup: dict):
        self.generate_train_test(train, test, lookup)
        self.fit()

    def evaluate_accuracy(self, evaluator:Evaluator):
        self.evaluate()
        return evaluator.get_accuracy(self.correct_count, self.len_test_data)
    
    def classify(self, evaluator: Evaluator, print_incorrect:bool = False):
        self.evaluate()
        #self.evaluate_clusters()

        if print_incorrect:
            evaluator.print_incorrect(self.incorrect)

        print("\nK-Means Classification Results")
        print("k:", self.k)

        evaluator.print_accuracy(self.correct_count, self.len_test_data)