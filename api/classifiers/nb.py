'''
This file contains the Naive Bayes algorithm for Pokelytics.
'''

class NaiveBayes:

    def __init__(self, data): #use pokemon_onehot_keep_type.csv here
        # Initalize data list
        self.data = data
        self.columns = []
        self.total_word_count = len(data)
        # Training Data (70% of total data)
        self.train_data = []
        self.train_data_wcnt = 0 
        # Classification Data (30% of total data)
        self.classify_data = []
        self.classify_data_wcnt = 0
        #Naive Bayes Algorithm
        self.unique_cats = {}
        self.unique_cats_prob = {}
        self.unique_cats_prob_xy = {}
        self.unique_cats_NB = {}

    #SOLID (DO NOT CHANGE)
    def set_columns(self): #sets the columns to sort the data for training
        for i, row in enumerate(self.data):
            self.columns.append(row)
            self.data.remove(row)
            if i > 176: break

    def split_data(self):
        # Split data into training and classification sets)
        split_index = int(0.7 * self.total_word_count)
        self.train_data = self.data[:split_index] # first 70%
        self.classify_data = self.data[split_index:] # other 30%
        self.train_data_wcnt = len(self.train_data)
        self.classify_data_wcnt = len(self.classify_data)
    
    #PENDING (UPDATE WHEN CHANGE AVAILABLE)
    def generate_training_data(self):
        pass

        
    def x_given_total(self, data):
        # P(X) = Count(X)/Count(Total)
        # For computing #P(Intrest)
        count = 0
        for i, row in enumerate(data):
            if i==0: continue
            count += 1
        return count/self.total_word_count
    
    def x_given_y(self, data, pOdata, smooth):
        # P(X|Y) = Count(X|Y)/Count(Y)
        # For computing #P(Intrest|Y)
        bow_temp = data
        count = 0
        for i, row in enumerate(bow_temp):
            if i==0: continue
            if row == :
                count += 1
        return (count + smooth)/(pOdata + (smooth * self.unique_words))

    def bayes_theorem(self, data):
        # P(Bayes(X)) = P(X|Cat)*P(X)/P(Cat)
        # For computing #P(Bayes(X))
        pass
    
    def display_results(self, data):
        print("RESULT OF BAYES ALGORITHM: ")
        int n = 0
        for cat, prob in self.unique_cats_NB.items():
            print(f"Pokemon {self.classify_data[n * 176]} Category: {cat}, Probability: {prob}")
            n += 1

    #MAIN
    def naive_bayes(self, data):
        #Ask for input for smoothing parameter
        k = float(input("Enter value of smoothing parameter: "))
        # Bag of Words
        self.generate_bow()
        # Naive Bayes Computations
        # self.x_given_total() --saved to self.unique_cats_prob
        # self.x_given_y() --saved to self.unique_cats_prob_xy
        # self.bayes_theorem() --saved to self.unique_cats_NB
        # self.display_results() --display results of Naive Bayes to frontend

        # TODO: 
        # Compute Bayes for each category pair (iterative loop)
        # Compute NB for all categories (func)
        # Comparison of NB classification to actual classification (func)
        # Group Categories by unique abilities and stats (func)
