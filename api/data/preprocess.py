'''
preprocess.py 
Holds all functions related to the preprocessing 
done on the original raw dataset (pokemon.csv)
'''

import csv
import re
import random

from data import Data

RAW_FILE = "raw_data/pokemon.csv"
CATEGORICAL_COLUMNS = ["type_1","ability_1","hidden_ability","color","shape"] 
MIN_FREQUENCY = 5
SEED = 10
TARGET = "type_1"
# Note: Target does not need to be One Hot Encoded for Naive Bayes and K-Nearest Neighbors
# Target MUST be One Hot Encoded for K-Clustering

class Preprocessor:
    def __init__(self):

        self.data_utils = Data()

        self.data = self.load_csv(RAW_FILE)

        self.n_rows = len(self.data)
        self.n_cols = len(self.data[0])  

        #self.raw_header = self.build_header_lookup(self.data)
        #self.unique_cats = self.get_unique_categories(self.data)
        #self.one_hot_data = self.one_hot_encoding(self.data)
        self.save_csv(self.shuffle_data(self.data), "pokemon_categorical.csv")
        #self.save_csv(self.shuffle_csv("pokemon_onehot_min5_keep_type.csv"), "pokemon_onehot_keep_type.csv")

    def build_header_lookup(self, data:list):
        header = {}
        for j, row in enumerate(data[0]):
                header[j] = row
        return header
 
    def load_csv(self, filename: str):
        with open(filename, newline='') as csvfile:
            data=[]
            pokemon = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(pokemon):
                row_clean = []
                for j in range(0, len(row)-1):
                    clean = row[j].lower()
                    clean = re.sub(r'[\s]', "", clean)
                    row_clean.append(clean)
                data.append(row_clean)
            return data

    def shuffle_csv(self, filename: str):
        random.seed(SEED)
        data = self.data_utils.load_csv(filename)
        random.shuffle(data)
        return data

    def shuffle_data(self, data: list):
        random.seed(SEED)
        random.shuffle(data)
        return data

    def get_unique_categories(self, data:list):
        counts = {}

        for i, row in enumerate(data):
            if i==0: continue
            for j, category in enumerate(row):

                curr_feature = self.raw_header[j]

                if curr_feature not in CATEGORICAL_COLUMNS:
                    continue

                counts.setdefault(curr_feature, {})
                counts[curr_feature][category] = counts[curr_feature].get(category, 0) + 1  # Count category frequencies per feature

        unique_categories = {}
        for features, cat in counts.items():
            if features == TARGET:                          # Do not bucket the target
                unique_categories[features] = list(cat)     
                continue
            kept = [c for c, n in cat.items() if n >= MIN_FREQUENCY]    # Keep only categories than are greater than or equal to the MIN_FREQUENCY
            if len(kept) < len(cat):    # If the categories we are keeping is less than the categories before, then we append other to the categories
                kept.append("other")
            unique_categories[features] = kept

        return unique_categories

    def one_hot_header(self, data: list):
        one_hot_header = []
        for feature in data[0]:
            if feature in CATEGORICAL_COLUMNS:
                one_hot_header.extend("is_" + feature + "_" + f for f in self.unique_cats[feature])
            else:
                one_hot_header.append(feature)
        return one_hot_header

    def one_hot_encoding(self, data: list):
        one_hot_data = []
        one_hot_data.append(self.one_hot_header(data))

        for i, row in enumerate(data):
            if i == 0:
                continue

            new_row = []
            for j, category_val in enumerate(row):
                curr_feature = self.raw_header[j]

                if curr_feature in CATEGORICAL_COLUMNS:
                    cats = self.unique_cats[curr_feature]                           # Get all unique categories for that feature
                    val = category_val if category_val in cats else "other"         # If the value is in cats, then the value is that category, else, it's other
                    for category in cats:
                        new_row.append(1 if val == category else 0)
                else:
                    new_row.append(category_val)
                    
            one_hot_data.append(new_row)

        return one_hot_data

    
    def save_csv(self, data: list, outfile: str):
        with open(outfile, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)

_preprocessor = Preprocessor()

# One Hot Encoding: https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/
# Grouping infrequent categories into an ‘other’ bucket: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# Sparse Data: https://www.analyticsvidhya.com/blog/2022/10/dealing-with-sparse-datasets-in-machine-learning/
