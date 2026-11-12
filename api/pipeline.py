from collections.abc import Callable

from services.classifiers.nb import NaiveBayes
from services.classifiers.knn import KNNClassifier
from services.classifiers.kmeans import KMeans

from services.data import Data
from pathlib import Path
from services.evaluator import Evaluator

DATA_DIR = Path(__file__).parent / 'data'
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

    def is_2d(self, data):
        return (
            isinstance(data, list)
            and len(data) > 0
            and all(isinstance(row, list) for row in data)
        )

    def grid_search(self):
        best_nb, score_nb = self.tune(
            self.cat_train,
            [{"smooth": s} for s in (0.1, 0.5, 0.35, 0.40, 0.46, 0.47, 0.45, 0.25, 0.65, 0.75, 1.0, 2.0)],
            NaiveBayes,
            NaiveBayes.evaluate,
            self.ev
        )
        print(f"NB Best: {best_nb}, {score_nb}")

        best_knn, score_knn = self.tune(
            self.onehot_train,
            [{"k": k} for k in (1, 3, 5, 7, 9, 11, 15, 21)],
            KNNClassifier,
            KNNClassifier.evaluate,
            self.ev,
            self.onehot_header
        )
        print(f"KNN Best: {best_knn}, {score_knn}")

        best_km, score_km = self.tune(
            self.onehot_train,
            [{"k": k} for k in (18*1, 18*2, 18*3, 18*4, 18*5, 18*6, 18*7)],
            KMeans,
            KMeans.evaluate,
            self.ev,
            self.onehot_header
        )
        print(f"KM Best: {best_km}, {score_km}")
 
    def run_best(self, verbose=False):
        print("\n------------------------------------------")
        self.ev.print_dataset_metrics(len(self.cat_data), len(self.cat_train), len(self.cat_test))

        print("\n------------------------------------------")
        self.nb = NaiveBayes(NB_BEST_S)
        self.nb.build(self.cat_train, self.cat_test)
        self.nb.classify(self.ev, verbose)

        print("\n------------------------------------------")
        self.knn = KNNClassifier(KNN_BEST_K)
        self.knn.build(self.onehot_train, self.onehot_test, self.onehot_header)
        self.knn.classify(self.ev, verbose)

        print("\n------------------------------------------")
        self.kmeans = KMeans(KMEANS_BEST_K)
        self.kmeans.build(self.onehot_train, self.onehot_test, self.onehot_header)
        self.kmeans.classify(self.ev, verbose)
        print("\n------------------------------------------")

    
    def tune(self, train_data: list, hyperparams: list, classifier: type,
                    metric: Callable, evaluator: Evaluator, *build_args, split: int = 5):

        folds = [train_data[i::split] for i in range(split)]
        results = []

        for n, params in enumerate(hyperparams, start=1):
            scores = []

            for i, val_fold in enumerate(folds):
                train_folds = [row for j, f in enumerate(folds) if j != i for row in f]

                model = classifier(**params)
                model.build(train_folds, val_fold, *build_args)
                eval_output = metric(model, evaluator)
                scores.append(eval_output["f1"])
                
            mean = sum(scores) / len(scores) if scores else 0
                
            results.append((params, mean))

        best_params, best_score = max(results, key=lambda r: r[1])
        
        return best_params, best_score

        #best_params, best_score = max(results, key=lambda r: r[1])
        #return best_params, best_score


p = Pipeline()


KMEANS_BEST_K = 108
KNN_BEST_K = 1
NB_BEST_S = 0.1

#p.grid_search()
p.run_best()

