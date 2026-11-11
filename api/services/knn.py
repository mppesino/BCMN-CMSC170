
import csv
from evaluator import Evaluator

PREFIXES = ("is_ability_", "is_shape_", "is_color_")

class KNNClassifier:

    #initialize the KNN classifier with k and p values
    def __init__(self, k:int=3, p:int=1):
        self.k = k
        self.p = p

        self.X_training = []
        self.Y_training = []

        self.X_test = []
        self.Y_test = []

        self.pokemon_names = []
        self.pokemon_ids = []

        self.len_data = 0
        self.len_train_data = 0
        self.len_test_data = 0

        self.evaluator = Evaluator()
            
    def generate_train_test(self, train: list, test: list, lookup: dict):
        # lookup is {"column_name": index}
        feature_idx = [i for name, i in lookup.items() if name.startswith(PREFIXES)]
        type_idx = lookup["type_1"]
        name_idx = lookup["name"]
        id_idx = lookup["pokedex_id"]

        for row in train:
            self.X_training.append([float(row[i]) for i in feature_idx])
            self.Y_training.append(row[type_idx])

        for row in test:
            self.X_test.append([float(row[i]) for i in feature_idx])
            self.Y_test.append(row[type_idx])
            self.pokemon_names.append(row[name_idx])
            self.pokemon_ids.append(row[id_idx])

        self.len_train_data = len(train)
        self.len_test_data = len(test)
            

    #compute the Minkowski distance between two points
    def minkowski(self, p1, p2, p):
        total = 0

        for i in range(len(p1)):
            total += abs(p1[i] - p2[i]) ** p

        return total ** (1 / p)


    #predict the type of a Pokemon based on the types of its K nearest neighbors
    #counts the occurrences of each type among the K nearest neighbors
    #returns the most common type as the predicted type for the given Pokemon
    def predict_type(self, neighbors):
        type_counts = {}

        for neighbor in neighbors:
            pokemon_type = neighbor[1]

            if pokemon_type in type_counts:
                type_counts[pokemon_type] += 1
            else:
                type_counts[pokemon_type] = 1

        most_common_type = None
        highest_count = 0

        for pokemon_type, count in type_counts.items():
            if count > highest_count:
                highest_count = count
                most_common_type = pokemon_type

        return most_common_type

    #helper function to get the distance from a neighbor tuple
    def get_distance(self, item):
        return item[0]

    #compute the K nearest neighbors for a given Pokemon and predict its type based on the types of its K nearest neighbors
    def knn(self, pokemon_to_predict, K, p):
        distances = []

        for i in range(len(self.X_training)):
            dist = self.minkowski(self.X_training[i], pokemon_to_predict, p)
            distances.append((dist, self.Y_training[i]))

        distances.sort(key=self.get_distance)
        neighbors = distances[:K]

        return self.predict_type(neighbors)

    #main function to run the KNN classifier, prompts the user for values of k and p, and computes the classification accuracy on the test set
    def classify(self, evaluator:Evaluator):

        correct = 0
        predictions = []
        incorrect_predictions = []

        for i in range(len(self.X_test)):
            prediction = self.knn(
                self.X_test[i],
                self.k,
                self.p
            )

            predictions.append(prediction)

            if prediction == self.Y_test[i]:
                correct += 1
            else:
                incorrect_predictions.append({
                    'id': self.pokemon_ids[i],
                    'pokemon': self.pokemon_names[i],
                    'actual': self.Y_test[i],
                    'predicted': predictions[i],
                })

        total = len(self.Y_test)

        evaluator.print_incorrect(incorrect_predictions)

        print("\nKNN Classification Results")
        print("k =", self.k)
        print("p =", self.p)

        evaluator.print_dataset_metrics(self.len_data, self.len_train_data, self.len_test_data)
        evaluator.print_accuracy(correct, total)

    def build(self, train:list, test:list, lookup:dict):
        self.generate_train_test(train, test, lookup)


