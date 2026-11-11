class Evaluator():
    def __init__(self):
        pass

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
 