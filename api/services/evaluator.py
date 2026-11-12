import numpy as np
from utils import Utils

class Evaluator():
    def __init__(self):
        self.utils = Utils()
        pass

    def evaluate_metrics(self, predictions:list):

        incorrect = []
        correct = []

        for item in predictions:
            if item["predicted"] != item["actual"]:
                incorrect.append(item)
            else:
                correct.append(item)

        mini_trix = self.get_type_mini_trix_dictionary(predictions)
        per_type_f1, means = self.get_mean_metrics(mini_trix)

        n_correct = len(correct)
        n_incorrect = len(incorrect)
        total = n_correct + n_incorrect

        accuracy = self.get_accuracy(n_correct, total)

        output = {
            "total": total,
            "correct":n_correct,
            "incorrect":n_incorrect,
            "acc": accuracy,
            "r": means["p"],
            "p": means["r"],
            "f1": means["f1"],
            "inc_arr": incorrect,
            "per_type_f1": per_type_f1
        }

        return output

    # Formats and Displays the 18 x 18 Matrix for each classifier in a Actual x Predicted format
    def get_type_confusion_matrix(self, predictions, pokemon_types=None, label="Pokemon Type"):
        if pokemon_types is None:
            pokemon_types = [
                "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
                "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
                "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"
            ]

        pokemon_types = [self.utils.normalize_type_name(ptype) for ptype in pokemon_types]

        matrix = {actual_type: {predicted_type: 0 for predicted_type in pokemon_types} for actual_type in pokemon_types}

        for pokemon in predictions:
            actual = self.utils.normalize_type_name(pokemon.get("actual"))
            predicted = self.utils.normalize_type_name(pokemon.get("predicted"))

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
    def get_type_mini_trix_dictionary(self, predictions, pokemon_types=None):
        if pokemon_types is None:
            pokemon_types = [
                "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
                "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
                "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"
            ]

        pokemon_types = [self.utils.normalize_type_name(ptype) for ptype in pokemon_types]
        mini_trix_list = []

        for pokemon_type in pokemon_types:
            tp = fn = fp = tn = 0

            for pokemon in predictions:
                actual = self.utils.normalize_type_name(pokemon.get("actual"))
                predicted = self.utils.normalize_type_name(pokemon.get("predicted"))
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

    def get_accuracy(self, correct, total):
        return (correct / total)*100
    
    def get_recall(self, matrix):
        tp = matrix[0][0]
        fn = matrix[0][1]

        if tp + fn == 0:
            return 0

        recall = tp / (tp + fn)
        return recall

    def get_precision(self, matrix):
        tp = matrix[0][0]
        fp = matrix[1][0]
       
        if tp + fp == 0:
            return 0

        precision = tp / (tp + fp)
        return precision

    def get_f1_score(self, recall, precision):

        total_f1_score = 0

        if recall + precision == 0:
            return 0

        total_f1_score = 2 * (precision * recall) / (precision + recall)

        return total_f1_score

    def get_mean_metrics(self, mini_trix_list):
        per_type = {}

        for entry in mini_trix_list:
            for pokemon_type, matrix in entry.items():
                recall = self.get_recall(matrix)
                precision = self.get_precision(matrix)
                f1 = self.get_f1_score(recall, precision)
                per_type[pokemon_type] = {"r": recall, "p": precision, "f1": f1}

        n = len(per_type)
        if n == 0:
            return per_type, {"r": 0, "p": 0, "f1": 0}

        means = {
            key: sum(scores[key] for scores in per_type.values()) / n
            for key in ("r", "p", "f1")
        }

        return per_type, means

    def print_accuracy(self, correct, total, accuracy):
        correct_lbl = "Correct"
        incorrect_lbl = "Incorrect"
        total_lbl = "Total"
        acc_lbl = "Accuracy"

        incorrect = total-correct

        print(f"\n{correct_lbl:<15}{incorrect_lbl:<15}{total_lbl:<15}{acc_lbl:<15}")
        print(f"{correct:<15}{incorrect:<15}{total:<15}{str(round(accuracy,2))+"%":<15}")

    def print_eval_metrics(self, f1_output, r, p, f1):
        recall_lbl = "Recall"
        precision_lbl = "Precision"
        f1_score_lbl = "F1 Score"

        '''
        for k in f1_output.keys():
            print(f"\n{k}:")
            print(f"\n{recall_lbl:<15}{precision_lbl:<15}{f1_score_lbl}")
            print(f"{round(f1_output[k]["r"],2):<15}{round(f1_output[k]["p"],2):<15}{round(f1_output[k]["f1"],2):<15}")
        '''

        #print(f"\nMean:")
        print(f"\n{recall_lbl:<15}{precision_lbl:<15}{f1_score_lbl}")
        print(f"{round(r,2):<15}{round(p,2):<15}{round(f1,2):<15}")

    def print_all_metrics(self, output):
        self.print_accuracy(output["correct"], output["total"], output["acc"])
        self.print_eval_metrics(output["per_type_f1"], output["r"], output["p"],  output["f1"])

    def print_incorrect(self, incorrect:list):
        id = "ID"
        pokemon_title = "Pokemon"
        predicted = "Predicted Type"
        actual = "Actual Type"
        print("\nIncorrect Classifications: \n")
        if incorrect:
            print(f"{id:<5}{pokemon_title:<30}{predicted:<30}{actual:<30}")
            print("-----------------------------------------------------------------------------------")
            for pokemon in incorrect:
                print(f"{pokemon['id']:<5}{pokemon['pokemon']:<30}{pokemon['predicted']:<30}{pokemon['actual']:<30}")
            print("-----------------------------------------------------------------------------------")
            print(f"{id:<5}{pokemon_title:<30}{predicted:<30}{actual:<30}")

        else:
            print("None")
        print("\nIncorrect Classifications ^ \n")

    def print_dataset_metrics(self, total_data, n_training_data, n_classify_data):
        print("\nDataset Metrics:\n")
        print(f"Total Data Count: {total_data}")
        print(f"Training Data Count: {n_training_data}")
        print(f"Classify Data Count: {n_classify_data}")
