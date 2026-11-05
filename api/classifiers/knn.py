
import csv

class KNNclassifier:

    #initialize the KNN classifier with k and p values
    def __init__(self, k, p):
        self.k = k
        self.p = p

        self.X_training = []
        self.Y_training = []

        self.X_test = []
        self.Y_test = []

    #load the data from the CSV file and convert the features to be used in the KNN algorithm in the form of a list of floats
    def load_data(self, filename):
        data = []

        with open(filename, "r") as file:
            reader = csv.DictReader(file)

            for row in reader:
                data.append(row)

        for row in data:
            for column in row:
                if column.startswith("is_ability_") or column.startswith("is_shape_") or column.startswith("is_color_"):
                    row[column] = float(row[column])

        return data

    #build the training and test sets from the loaded data
    def build_training_and_test_sets(self, data):

        training_data = []
        test_data = []

        #split the data into training and test sets (70% training, 30% testing)
        total_training_data = int(len(data) * 0.70)

        for i in range(len(data)):
            if i < total_training_data:
                training_data.append(data[i])
            else:
                test_data.append(data[i])

        #build the training set
        for row in training_data:
            pokemon_features = []

            for column in row:
                if column.startswith("is_ability_") or column.startswith("is_shape_") or column.startswith("is_color_"):
                    pokemon_features.append(row[column])

            self.X_training.append(pokemon_features)
            self.Y_training.append(row["type_1"])

        #build the test set from the remaining data
        for row in test_data:
            pokemon_features = []

            for column in row:
                if column.startswith("is_ability_") or column.startswith("is_shape_") or column.startswith("is_color_"):
                    pokemon_features.append(row[column])

            self.X_test.append(pokemon_features)
            self.Y_test.append(row["type_1"])

        print("\nData set loaded successfully.")
        print("Total numbers of records:", len(data))
        print("Training instances:", len(training_data))
        print("Testing instances:", len(test_data))

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
    def run_knn(self):

        while True:
            k = int(input("Enter an odd value of k: "))

            if k % 2 != 0:
                break
            else:
                print("Please enter an ODD value.")

        while True:
            p = int(input("Enter the value of p for Minkowski distance: "))

            if p > 0:
                break
            else:
                print("Please enter a positive value.")

        correct = 0
        predictions = []
        incorrect_predictions = []

        for i in range(len(self.X_test)):
            prediction = self.knn(
                self.X_test[i],
                k,
                p
            )

            predictions.append(prediction)

            if prediction == self.Y_test[i]:
                correct += 1
            else:
                incorrect_predictions.append(i)

        total = len(self.Y_test)
        incorrect = total - correct
        accuracy = (correct / total) * 100

        print("\nKNN Classification Results")
        print("k =", k)
        print("p =", p)
        print("\nCorrect Predictions:", correct)
        print("Incorrect Predictions:", incorrect)
        print("Classification Accuracy:", accuracy, "%")

        print("\nIncorrect Classifications:")

        for i in incorrect_predictions:
            print("Actual:", self.Y_test[i], "Predicted:", predictions[i])


classifier = KNNclassifier(0, 0) #initialize the KNN classifier with k and p values of 0, which will be set later by the user

data = classifier.load_data("BCMN-CMSC170/api/data/pokemon_onehot_keep_type.csv") #open the CSV file and load the data into a list of dictionaries

classifier.build_training_and_test_sets(data) #build the training and test sets from the loaded data

classifier.run_knn() #perform the KNN classification on the test set and print the results
