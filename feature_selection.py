import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import random

class feature_selection:
    def __init__(self, filename):
        self.data = pd.read_csv(filename, header=None)
        self.n = self.data.shape[1]-1
        self.num_objs = 2
        self.target_functions = np.array([0,0])


    def generate_initial_population(self, population_size):
        population = np.empty(population_size, dtype=object)
        min_num_selected_features = 5
        max_num_selected_features = 30
        for i in range(population_size):

            num_elements = random.randint(min_num_selected_features,max_num_selected_features-1)
            sequence_items = set(random.sample(range(0, self.n), num_elements))

            if not sequence_items in population:
                population[i] = sequence_items
            else:
                population[i] = set(random.sample(range(0,self.n), num_elements))

            #print(population[i])

        return population

    def objective_function(self, representation):

        population_size = representation.shape[0]
        clf = LinearDiscriminantAnalysis()
        evaluations = np.zeros((population_size, 2))

        for i in range(population_size):
            X = self.data.iloc[:,list(representation[i])]
            y = self.data.iloc[:,self.n]
            num_features = len(representation[i])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

            #print(representation[i])

            clf.fit(X_train,y_train)

            evaluations[i][0] = 1 - clf.score(X_test, y_test)
            evaluations[i][1] = (num_features - 1) / (self.n - 1)
        return evaluations

    def get_name(self):
        return 'Feature Extraction'


