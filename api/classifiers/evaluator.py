import numpy as np


class Evaluator():
    def __init__(self):
        pass
    
    # Use this for Display Shortening so it dosen't clip and cook the terminal output
    # Might be useful for text display in the future
    def normalize_type_name(self, value):
        if value is None:
            return ""
        return str(value).strip().lower().capitalize()

    def shorten_type_name(self, value, max_len=3):
        name = self.normalize_type_name(value)
        return name[:max_len] if name else ""


    def print_incorrect(self, incorrect:list):
        id = "ID"
        pokemon_title = "Pokemon"
        actual = "Predicted Type"
        predicted = "Actual Type"
        print("\nIncorrect Classifications: \n")
        if incorrect:
            print(f"{id:<5}{pokemon_title:<30}{predicted:<30}{actual:<30}")
            print("-----------------------------------------------------------------------------------")
            for pokemon in incorrect:
                print(f"{pokemon['id']:<5}{pokemon['pokemon']:<30}{pokemon['actual']:<30}{pokemon['predicted']:<30}")
            print("-----------------------------------------------------------------------------------")
            print(f"{id:<5}{pokemon_title:<30}{predicted:<30}{actual:<30}")

        else:
            print("None")
        print("\nIncorrect Classifications ^ \n")

    def print_dataset_metrics(self, total_data, n_training_data, n_classify_data):
        print("\nDataset Metrics:\n")
        print(f"Total Data Count: {total_data}")
        print(f"Training Data Count: {n_training_data}")
        print(f"Classify Data Count: {n_classify_data}\n")

    def get_accuracy(self, correct, total):
        return (correct / total)*100

    def print_accuracy(self, correct, total):
        print("\nAccuracy:\n")
        correct_lbl = "Correct"
        incorrect_lbl = "Incorrect"
        total_lbl = "Total"
        acc_lbl = "Accuracy"

        incorrect = total-correct

        print(f"{correct_lbl:<15}{incorrect_lbl:<15}{total_lbl:<15}{acc_lbl:<15}")
        print(f"{correct:<15}{incorrect:<15}{total:<15}{str(round(self.get_accuracy(correct,total),2))+"%":<15}")

    # Formats and Displays the 18 x 18 Matrix for each classifier in a Actual x Predicted format
    def save_type_confusion_matrix(self, predictions, pokemon_types=None, label="Pokemon Type"):
        if pokemon_types is None:
            pokemon_types = [
                "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
                "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
                "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"
            ]

        pokemon_types = [self.normalize_type_name(ptype) for ptype in pokemon_types]

        matrix = {actual_type: {predicted_type: 0 for predicted_type in pokemon_types} for actual_type in pokemon_types}

        for pokemon in predictions:
            actual = self.normalize_type_name(pokemon.get("actual"))
            predicted = self.normalize_type_name(pokemon.get("predicted"))

            if actual in matrix and predicted in matrix[actual]:
                matrix[actual][predicted] += 1
            elif actual and predicted:
                if actual not in matrix:
                    matrix[actual] = {predicted_type: 0 for predicted_type in pokemon_types}
                if predicted not in matrix[actual]:
                    matrix[actual][predicted] = 0
                matrix[actual][predicted] += 1

                for existing_type in pokemon_types:
                    matrix.setdefault(existing_type, {p: 0 for p in pokemon_types})
                    matrix[existing_type].setdefault(predicted, 0)

        matrix_array = np.array(
            [[matrix[actual_type].get(predicted_type, 0) for predicted_type in pokemon_types] for actual_type in pokemon_types],
            dtype=int
        )

        short_labels = [self.shorten_type_name(ptype) for ptype in pokemon_types]
        print(f"\n{label} confusion matrix")
        print("      " + " ".join(f"{name:>3}" for name in short_labels))
        for i, actual_type in enumerate(short_labels):
            row = matrix_array[i]
            print(f"{actual_type:>3} " + " ".join(f"{value:>3}" for value in row))

        return matrix_array


    # Display Func for the 2x2 Matrixes
    def print_type_mini_trix_dictionary(self, mini_trix_list):
        print("\nMini-trix:")
        for entry in mini_trix_list:
            for pokemon_type, matrix in entry.items():
                print(f"\n{pokemon_type}:")
                print(np.array(matrix, dtype=int))

    # Gets the 2x2 Matrix of each pokemon type, instead of the 18x18 matrix u can just use the same source but just make it more detailed tbh 
    def save_type_mini_trix_dictionary(self, predictions, pokemon_types=None):
        if pokemon_types is None:
            pokemon_types = [
                "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
                "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
                "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"
            ]

        pokemon_types = [self.normalize_type_name(ptype) for ptype in pokemon_types]
        mini_trix_list = []

        for pokemon_type in pokemon_types:
            tp = fn = fp = tn = 0

            for pokemon in predictions:
                actual = self.normalize_type_name(pokemon.get("actual"))
                predicted = self.normalize_type_name(pokemon.get("predicted"))
                is_actual = actual == pokemon_type
                is_predicted = predicted == pokemon_type

                if is_actual and is_predicted:
                    tp += 1
                elif is_actual and not is_predicted:
                    fn += 1
                elif not is_actual and is_predicted:
                    fp += 1
                else:
                    tn += 1

            mini_trix_list.append({pokemon_type: [[tp, fn], [fp, tn]]})

        return mini_trix_list

    def save_type_dictionary(self, predictions, pokemon_types=None):
        return self.save_type_mini_trix_dictionary(predictions, pokemon_types)
    
    def recall(self, matrix):
        tp = matrix[0][0]
        fn = matrix[0][1]

        if tp + fn == 0:
            return 0

        recall = tp / (tp + fn)
        return recall

    def precision(self, matrix):
        tp = matrix[0][0]
        fp = matrix[1][0]

       

        if tp + fp == 0:
            return 0

        precision = tp / (tp + fp)
        return precision

    def f1_score(self, recall, precision):

        total_f1_score = 0

        if recall + precision == 0:
            return 0

        total_f1_score = 2 * (precision * recall) / (precision + recall)

        return total_f1_score


    def f1_mean(self, mini_trix_list):
        f1_scores = []
        total_f1_score = 0

        for entry in mini_trix_list:
            for pokemon_type, matrix in entry.items():
                recall = self.recall(matrix)
                precision = self.precision(matrix)
                f1_score = self.f1_score(recall, precision)
                f1_scores.append(f1_score)
                total_f1_score += f1_score

        if len(f1_scores) == 0:
            return 0, f1_scores

        mean_f1_score = total_f1_score / len(f1_scores)

        return mean_f1_score, f1_scores

    def print_eval_metrics(self, mini_trix_list):
        print("\nEvaluation Metrics:")
        for entry in mini_trix_list:
            for pokemon_type, matrix in entry.items():
                recall = self.recall(matrix)
                precision = self.precision(matrix)
                f1_score = self.f1_score(recall, precision)

                print(f"\n{pokemon_type}:")
                print("Recall: ", recall)
                print("Precision: ", precision)
                print("F1 Score: ", f1_score)

        print("\nMean F1 Score:", self.f1_mean(mini_trix_list)[0])
        mean_f1_score, f1_scores = self.f1_mean(mini_trix_list)

        return mean_f1_score, f1_scores