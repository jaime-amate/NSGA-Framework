import numpy as np
import random
from deap import benchmarks
import dask
import time

def compute_evaluation(representation):
    time.sleep(1)
    return list(benchmarks.zdt1(representation))

class ZDT1:
    num_objs = 2
    target_functions = np.zeros(num_objs, dtype=np.int)
    n = 30
    lower_bound = 0.0
    upper_bound = 1.0
    datatype = np.float64

    def generate_initial_population(self, population_size):
        return np.random.rand(population_size, self.n)

    def objective_function(self,representation):
        num_individuals = len(representation)
        evaluations = np.zeros((num_individuals, self.num_objs))

        #sequential
        for i in np.arange(num_individuals):
            #time.sleep(1)
            evaluations[i] = list(benchmarks.zdt1(representation[i]))

        '''
        #parallel
        delayed_results = [dask.delayed(compute_evaluation)(representation[i]) for i in range(num_individuals)]
        evaluations = dask.compute(*delayed_results)
        evaluations = np.asarray(evaluations)
        '''
        return evaluations



    def pareto_front(self,x):
        return 1 - np.sqrt(x)

    def get_name(self):
        return 'ZDT1'

class ZDT2:
    num_objs = 2
    target_functions = np.zeros(num_objs, dtype=np.int)
    n = 30
    lower_bound = 0.0
    upper_bound = 1.0
    datatype = np.float64

    def generate_initial_population(self, population_size):
        return np.random.rand(population_size, self.n)

    def objective_function(self,representation):
        num_individuals = len(representation)
        evaluations = np.zeros((num_individuals, self.num_objs))

        for i in np.arange(num_individuals):
            evaluations[i] = list(benchmarks.zdt2(representation[i]))

        return evaluations

    def pareto_front(self,x):
        return 1 - np.power(x,2)

    def get_name(self):
        return 'ZDT2'

class ZDT3:
    num_objs = 2
    target_functions = np.zeros(num_objs, dtype=np.int)
    n = 30
    lower_bound = 0.0
    upper_bound = 1.0
    datatype = np.float64

    def generate_initial_population(self, population_size):
        return np.random.rand(population_size, self.n)

    def objective_function(self,representation):
        num_individuals = len(representation)
        evaluations = np.zeros((num_individuals, self.num_objs))

        for i in np.arange(num_individuals):
            evaluations[i] = list(benchmarks.zdt3(representation[i]))

        return evaluations

    def pareto_front(self,x):
        return 0

    def get_name(self):
        return 'ZDT3'

class ZDT4:
    num_objs = 2
    target_functions = np.zeros(num_objs, dtype=np.int)
    n = 10
    lower_bound = 0.0
    upper_bound = 1.0
    datatype = np.float64

    def generate_initial_population(self, population_size):
        a = np.random.uniform(-5, 5, size=(population_size, self.n))

        for i in range(population_size):
            a[i][0] = random.random()

        print(a)
        return a

    def objective_function(self, representation):
        num_individuals = len(representation)
        evaluations = np.zeros((num_individuals, self.num_objs))

        for i in np.arange(num_individuals):
            evaluations[i] = list(benchmarks.zdt4(representation[i]))

        return evaluations

    def pareto_front(self, x):
        return 1 - np.sqrt(x)

    def get_name(self):
        return 'ZDT4'

class ZDT6:
    num_objs = 2
    target_functions = np.zeros(num_objs, dtype=np.int)
    n = 10
    lower_bound = 0.0
    upper_bound = 1.0
    datatype = np.float64

    def generate_initial_population(self, population_size):
        return np.random.rand(population_size, self.n)

    def objective_function(self, representation):
        num_individuals = len(representation)
        evaluations = np.zeros((num_individuals, self.num_objs))

        for i in np.arange(num_individuals):
            evaluations[i] = list(benchmarks.zdt6(representation[i]))

        return evaluations

    def pareto_front(self, x):
        return 1 - np.power(x, 2)

    def get_name(self):
        return 'ZDT6'

class DTLZ1:
    def __init__(self, num_objs):
        self.num_objs = num_objs
        self.target_functions = np.zeros(num_objs, dtype=np.int)
        self.k = 5
        self.n = num_objs + self.k -1
        self.lower_bound = 0.0
        self.upper_bound = 1.0

    def generate_initial_population(self, population_size):
        return np.random.rand(population_size, self.n)

    def objective_function(self, representation):
        num_individuals = len(representation)
        evaluations = np.zeros((num_individuals, self.num_objs))

        for i in range(num_individuals):
            evaluations[i] = list(benchmarks.dtlz1(representation[i], self.num_objs))

        return evaluations

    def get_name(self):
        return 'DTLZ1'

class DTLZ2:
    def __init__(self, num_objs):
        self.num_objs = num_objs
        #self.target_functions = np.ones(num_objs, dtype=np.int)
        self.target_functions = np.zeros(num_objs, dtype=np.int)
        self.k = 10
        self.n = self.num_objs

    def generate_initial_population(self, population_size):
        return np.random.rand(population_size, self.n)

    def objective_function(self,representation):
        num_individuals = len(representation)
        evaluations = np.zeros((num_individuals, self.num_objs))

        for i in range(num_individuals):
            evaluations[i] = list(benchmarks.dtlz2(representation[i], self.num_objs))

        return evaluations

    def get_name(self):
        return 'DTLZ2'

class DTLZ3:
    def __init__(self, num_objs):
        self.num_objs = num_objs
        self.target_functions = np.ones(num_objs, dtype=np.int)
        self.k = 10
        self.n = self.num_objs

    def generate_initial_population(self, population_size):
        return np.random.rand(population_size, self.n)

    def objective_function(self,representation):
        num_individuals = len(representation)
        evaluations = np.zeros((num_individuals, self.num_objs))

        for i in range(num_individuals):
            evaluations[i] = list(benchmarks.dtlz3(representation[i], self.num_objs))

        return evaluations

    def get_name(self):
        return 'DTLZ3'

class DTLZ4:
    def __init__(self, num_objs):
        self.num_objs = num_objs
        self.target_functions = np.ones(num_objs, dtype=np.int)
        self.k = 10
        self.n = self.num_objs

    def generate_initial_population(self, population_size):
        return np.random.rand(population_size, self.n)

    def objective_function(self,representation):
        num_individuals = len(representation)
        evaluations = np.zeros((num_individuals, self.num_objs))

        for i in range(num_individuals):
            evaluations[i] = list(benchmarks.dtlz4(representation[i], self.num_objs))

        return evaluations

    def get_name(self):
        return 'DTLZ4'