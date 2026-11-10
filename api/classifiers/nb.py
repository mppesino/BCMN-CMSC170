import csv
from pathlib import Path
from evaluator import Evaluator

'''
This file contains the Naive Bayes algorithm for Pokelytics.
'''

class NaiveBayes:

    def __init__(self, data=None, smooth=1.0): #use pokemon_onehot_keep_type.csv here
        # Initalize data list
        self.data = data
        self.columns = []
        self.total_word_count = len(data) if data is not None else 0
        # Training Data (70% of total data)
        self.train_data = []
        self.train_data_wcnt = 0
        # Classification Data (30% of total data)
        self.classify_data = []
        self.classify_data_wcnt = 0
        # Naive Bayes Algorithm
        self.unique_cats = {}
        self.unique_cats_prob = {}
        self.unique_cats_prob_xy = {}
        self.unique_cats_NB = {}
        # Extra Variables
        self.unique_words = []
        self.category_word_totals = {}
        self.class_counts = {}
        self.smooth = smooth

        self.evaluator = Evaluator()

        if self.data is None:
            self.generate_csv()

    def generate_csv(self):
        base_dir = Path(__file__).resolve().parents[2]
        csv_path = base_dir / 'api' / 'data' / 'pokemon_categorical_binning.csv'
        with open(csv_path, newline='', encoding='utf-8') as csv_file:
            self.data = list(csv.reader(csv_file))
        self.total_word_count = len(self.data)
        self.train_data = []
        self.classify_data = []
        self.train_data_wcnt = 0
        self.classify_data_wcnt = 0
        return self.data

    def set_columns(self): #sets the columns to sort the data for training
        for i, row in enumerate(self.data):
            self.columns.append(row)
            self.data.remove(row)
            if i > 145: break

    def split_data(self):
        # Split data into training and classification sets)
        split_index = int(0.7 * self.total_word_count)
        self.train_data = self.data[:split_index] # first 70%
        self.classify_data = self.data[split_index:] # other 30%
        self.train_data_wcnt = len(self.train_data)
        self.classify_data_wcnt = len(self.classify_data)

    def generate_cats(self):
        # Generate unique categories from training data
        self.unique_cats = {}
        for row in self.train_data:
            if not row or row[0] == 'pokedex_id':
                continue
            category = str(row[2]).strip().lower()
            if category not in self.unique_cats:
                self.unique_cats[category] = {}

    def get_words(self, row):
        #fetches the "words" that have value not "na"
        words = []
        if not row:
            return words
        for value in row[3:]:
            word = str(value).strip().lower()
            if word and word != 'na':
                words.append(word)
        return words

    def generate_training_data(self):
        # generates the training data for naive bayes
        self.generate_cats()
        for row in self.train_data:
            if not row or row[0] == 'pokedex_id':
                continue
            category = str(row[2]).strip().lower()
            for word in self.get_words(row):
                self.unique_cats[category][word] = self.unique_cats[category].get(word, 0) + 1
        self.unique_words = sorted({word for counts in self.unique_cats.values() for word in counts})
        self.category_word_totals = {cat: sum(counts.values()) for cat, counts in self.unique_cats.items()}
        self.class_counts = {}
        for row in self.train_data:
            if not row or row[0] == 'pokedex_id':
                continue
            category = str(row[2]).strip().lower()
            self.class_counts[category] = self.class_counts.get(category, 0) + 1
        return self.unique_cats

    def generate_bow(self):
        # generates the unique word list for each category ## computations for probability is also included here
        self.generate_training_data()
        self.x_given_total(self.train_data)
        self.x_given_y(self.train_data, self.category_word_totals, self.smooth)
        return self.unique_words

    def x_given_total(self, data):
        # P(X) = Count(X)/Count(Total)
        # For computing #P(Intrest)
        self.unique_cats_prob = {}
        for cat, word_counts in self.unique_cats.items():
            total = sum(word_counts.values())
            if total == 0:
                total = 1
            self.unique_cats_prob[cat] = {word: count / total for word, count in word_counts.items()}
        return self.unique_cats_prob

    def x_given_y(self, data, pOdata, smooth):
        # P(X|Y) = Count(X|Y)/Count(Y)
        # For computing #P(Intrest|Y)
        self.unique_cats_prob_xy = {}
        vocab_size = max(len(self.unique_words), 1)
        for cat, word_counts in self.unique_cats.items():
            category_total = pOdata.get(cat, 0)
            self.unique_cats_prob_xy[cat] = {}
            for word in self.unique_words:
                count = word_counts.get(word, 0)
                self.unique_cats_prob_xy[cat][word] = (count + smooth) / (category_total + (smooth * vocab_size))
        return self.unique_cats_prob_xy

    def bayes_theorem(self, data):
        # P(Bayes(X)) = P(X|Cat)*P(X)/P(Cat)
        # For computing #P(Bayes(X))
        if not data:
            return {}
        probabilities = {}
        total = 0.0
        for cat in self.unique_cats:
            prior = self.class_counts.get(cat, 0) / max(self.train_data_wcnt, 1)
            likelihood = 1.0
            vocab_size = max(len(self.unique_words), 1)
            for word in data:
                category_total = self.category_word_totals.get(cat, 0)
                conditional = self.unique_cats_prob_xy.get(cat, {}).get(word, (self.smooth) / (category_total + (self.smooth * vocab_size)))
                likelihood *= conditional
            probabilities[cat] = prior * likelihood
            total += probabilities[cat]

        if total == 0:
            self.unique_cats_NB = {cat: 0.0 for cat in probabilities}
            return self.unique_cats_NB

        self.unique_cats_NB = {cat: prob / total for cat, prob in probabilities.items()}
        return self.unique_cats_NB

    #MAIN
    def naive_bayes(self, data=None):
        if data is not None:
            self.data = data
            self.total_word_count = len(data)

        if not self.train_data and not self.classify_data:
            self.split_data()

        self.generate_bow()

        total_predictions = 0
        correct_predictions = 0
        incorrect_predictions = []

        for row in self.classify_data:
            if not row or row[0] == 'pokedex_id':
                continue
            row_words = self.get_words(row)
            probabilities = self.bayes_theorem(row_words)
            self.unique_cats_NB = probabilities

            predicted = max(probabilities, key=probabilities.get)
            actual = str(row[2]).strip().lower()
            is_correct = predicted == actual
            total_predictions += 1
            if is_correct:
                correct_predictions += 1
            else:
                incorrect_predictions.append({
                    'id': row[0].strip(),
                    'pokemon': row[1].strip(),
                    'actual': actual,
                    'predicted': predicted,
                })

        self.evaluator.print_incorrect(incorrect_predictions)

        self.evaluator.print_dataset_metrics(self.total_word_count-1,self.train_data_wcnt-1, self.classify_data_wcnt)
        print(f"Smoothing Factor: {self.smooth}\n")
        self.evaluator.print_accuracy(correct_predictions, total_predictions)

if __name__ == "__main__":
    try:
        smoothing = float(input("Enter smoothing factor: "))
    except ValueError:
        smoothing = 1.0

    model = NaiveBayes(smooth=smoothing)
    model.naive_bayes()
