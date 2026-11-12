import csv
from pathlib import Path

import numpy as np
from ..evaluator import Evaluator

'''
This file contains the Naive Bayes algorithm for Pokelytics.
'''

class NaiveBayes:

    def __init__(self, smooth=1.0):
        self.smooth = smooth

        # Naive Bayes Algorithm
        self.unique_cats = {}
        self.unique_cats_prob = {}
        self.unique_cats_prob_xy = {}
        self.unique_cats_NB = {}
        
        # Extra Variables
        self.unique_words = []
        self.category_word_totals = {}
        self.class_counts = {}

    def generate_cats(self):
        # Generate unique categories from training data
        self.unique_cats = {}
        for i, row in enumerate(self.train_data):
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
        for i, row in enumerate(self.train_data):
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

    def predict(self, x):
        probabilities = self.bayes_theorem(self.get_words(x))
        return max(probabilities, key=probabilities.get)

    def evaluate(self, evaluator: Evaluator):
        predictions = []

        for row in self.test_data:

            predictions.append({
                "id": row[0].strip(),
                "pokemon": row[1].strip(),
                "actual": str(row[2]).strip().lower(),
                "predicted": self.predict(row),
            })

        return evaluator.evaluate_metrics(predictions)

    def classify(self, evaluator:Evaluator, print_incorrect:bool=False):

        output = self.evaluate(evaluator)

        if print_incorrect:
            evaluator.print_incorrect(output["inc_arr"])

        print("\nNaive Bayes Classification Results")
        print(f"Smoothing Factor: {self.smooth}")
        
        evaluator.print_all_metrics(output)
        return output
    def build(self, train:list, test:list):

        # Training Data (70% of total data)
        self.train_data = train
        self.train_data_wcnt = len(self.train_data)

        # Classification Data (30% of total data)
        self.test_data = test
        self.test_data_wcnt = 0

        self.generate_bow()

