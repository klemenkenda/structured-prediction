# import preprocessing tools
from sklearn import preprocessing
import numpy as np
import pandas as pd

# import learning/evaluation
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# typing
from typing import Dict, List, Callable, Any, Tuple, Optional, \
    Counter as CounterType, Set

# FASTENER specific imports
from fastener.random_utils import shuffle
from fastener.item import Result, Genes, RandomFlipMutationStrategy, RandomEveryoneWithEveryone, \
    IntersectionMating, UnionMating, IntersectionMatingWithInformationGain, \
    IntersectionMatingWithWeightedRandomInformationGain
from fastener import fastener

# EXPERIMENT specific imports
from data import *

def make_target(df, h):
    y = df.shift(periods=-h, freq='1H')["pc"]
    #df.pop('pc15')
    #df.pop('pc30')
    #df.pop('pc45')
    y = y[h:]
    X = df[:-h]
    return X, y

# define evaluation function
def evaluation_fun(model: Any, genes: "Genes", shuffle_indices: Optional[List[int]] = None) -> "Result":
    test_data = XX_test[:, genes]
    if shuffle_indices:
        test_data = test_data.copy()
        for j in shuffle_indices:
            shuffle(test_data[:, j])
    pred = model.predict(test_data)
    res = Result(r2_score(labels_test, pred))
    return res

# run a series of experiments
# horizons = [1, 3, 6, 12, 24, 36]
# sensors = [312, 386, 235, 368, 290, 1]
horizons = [1]
sensors = [312]

for s in sensors:
    for h in horizons:
        output = "s{}h{}".format(s, h)
        print("\n\n *** {} *** \n".format(output))


        # load data
        # load data
        static_df = load_static()
        weather_df = pd.read_pickle('data/weather/weather.pkl')
        # load sensor - last parameter is id
        final_df = make_final(static_df, weather_df, s)

        X_df, y_df = make_target(final_df, h)

        # basic dataset split
        n_sample = X_df.shape[0]
        n_test = int(n_sample * 0.8)

        labels_train = y_df.to_numpy(dtype='float')[:n_test]
        labels_test = y_df.to_numpy(dtype='float')[n_test:]

        XX_train = X_df.to_numpy()[:n_test, :]
        XX_test = X_df.to_numpy()[n_test:, :]

        number_of_genes = XX_train.shape[1]
        general_model = DecisionTreeRegressor
        # general_model = LinearRegression
        # output folder name must be changed every time the algorithm is run
        output_folder_name = output

        # to start the algorithm initial_genes or initial_population must be provided
        initial_genes = [
            [398,]
        ]

        # Select mating selection strategie (RandomEveryoneWithEveryone, NoMating) and mating strategy
        # (UnionMating, IntersectionMating, IntersectionMatingWithInformationGain,
        # IntersectionMatingWithWeightedRandomInformationGain)
        # If regression model is used IntersectionMatingWithInformationGain, IntersectionMatingWithWeightedRandomInformationGain
        # must have regression=True set (eg. IntersectionMatingWithInformationGain(regression=True))
        mating = RandomEveryoneWithEveryone(pool_size=3, mating_strategy=IntersectionMatingWithWeightedRandomInformationGain(regression=True))

        # Random mutation (probability of gene mutating: 1 / number_of_genes)
        mutation = RandomFlipMutationStrategy(1 / number_of_genes)

        entropy_optimizer = fastener.EntropyOptimizer(
            general_model, XX_train, labels_train, evaluation_fun,
            number_of_genes, mating, mutation, initial_genes=initial_genes,
            config=fastener.Config(output_folder=output_folder_name, random_seed=2020, reset_to_pareto_rounds=5, number_of_rounds=1000)
        )
        # change number of rounds to 1000 for final evaluation
        entropy_optimizer.mainloop()