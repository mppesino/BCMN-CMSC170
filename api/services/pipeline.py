from collections.abc import Callable

from nb import NaiveBayes
from knn import KNNClassifier
from kmeans import KMeans

from data import Data
from pathlib import Path
from evaluator import Evaluator

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'api' / 'data'
CAT_BINNING = DATA_DIR / 'pokemon_categorical_binning.csv'
ONEHOT_KEEP_TYPE = DATA_DIR / 'pokemon_onehot_keep_type.csv'


class Pipeline:

    def __init__(self):
        self.dt = Data()
        self.ev = Evaluator()
        
        self.cat_data = self.dt.load_csv(CAT_BINNING)
        self.onehot_data = self.dt.load_csv(ONEHOT_KEEP_TYPE)

        self.cat_header = self.dt.build_header_lookup(self.cat_data)
        self.onehot_header = self.dt.build_index_lookup(self.onehot_data)

        self.cat_train, self.cat_test= self.split_data(self.cat_data)
        self.onehot_train, self.onehot_test  =  self.split_data(self.onehot_data)

        print("\n------------------------------------------")
        self.ev.print_dataset_metrics(len(self.cat_data), len(self.cat_train), len(self.cat_test))

        print("\n------------------------------------------")
        self.nb = NaiveBayes()
        self.nb.build(self.cat_train, self.cat_test)
        self.nb.classify(self.ev)

        print("\n------------------------------------------")
        self.knn = KNNClassifier()
        self.knn.build(self.onehot_train, self.onehot_test, self.onehot_header)
        self.knn.classify(self.ev)

        print("\n------------------------------------------")
        self.kmeans = KMeans()
        self.kmeans.build(self.onehot_train, self.onehot_test, self.onehot_header)
        self.kmeans.classify(self.ev)
        print("\n------------------------------------------")

    def split_data(self, data, train_pct=0.70):

        n_train = int(len(data) * train_pct)
        train_data = []
        test_data = []

        for i in range(len(data)):
            if i == 0:
                continue
            if i < n_train:
                train_data.append(data[i])
            else:
                test_data.append(data[i])

        return train_data, test_data

    def grid_search(self, train_data: list, algorithm:Callable, evaluate:Callable, split:int=3, *args, **kwargs):

        folds = [train_data[i::split] for i in range(split)]
        print(folds)

_pipeline = Pipeline()