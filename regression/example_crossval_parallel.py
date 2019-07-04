# Copyright (C) 2019 Peter Sarvari, University of Southern California
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import pandas as pd
import numpy as np
import importlib
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
import random
import time
import pickle
from keras.optimizers import Adam
from keras.models import model_from_json
from collections import Counter
import math
import concurrent.futures
from itertools import repeat

def get_performance_vals(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    stdev = np.std(y_test)
    var = stdev**2
    total_mse = var
    rsq = (total_mse - mse) / total_mse

    print("MSE: %.3f" %mse)
    print("R-squared is %.3f" %rsq)

    return mse, rsq


def do_kfolds(x, y, l):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    rsqs = []

    build_sequential_model = getattr(__import__(l[0]), "build_sequential_model")
    fit_model_batch = getattr(__import__(l[0]), "fit_model_batch")
    print("Preparing training and label data...OK")
    print("")

    for train, test in kfold.split(x, y):

        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        mean = np.mean(x_train, axis = 0)
        std = np.std(x_train, axis = 0)
        x_train -= mean
        x_train /= std

        x_test -= mean
        x_test /= std

        print("Training model on data...")
        s_training = time.time()
        M = build_sequential_model(l[1],l[2],shape = x_train.shape[1])
        trained_M = fit_model_batch(M, x_train, y_train, num_epoch=2000)

        print("Predicting data...")
        s_classify = time.time()
        y_pred = trained_M.predict(x_test)
        y_pred = np.array(y_pred)
        y_pred = y_pred.ravel()

        e_classify = time.time()
        print("Predicting data...OK, took: " + str((e_classify - s_classify)))

        mse, rsq = get_performance_vals(y_test, y_pred)

        rsqs.append(rsq)

    return trained_M, mean, std, np.mean(rsqs)

def crossvalfunc(x, y, parameters, train, test):

    x = np.array(x)
    y = np.array(y)

    trained_Ms = []
    means = []
    stds = []
    meanrsqs = []
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]


    for l in parameters:

        trained_M, mean, std, m_rsq = do_kfolds(x_train,y_train,l)
        trained_Ms.append(trained_M)
        means.append(mean)
        stds.append(std)
        meanrsqs.append(m_rsq)

    best_index = np.argmax(meanrsqs)
    chosen_model = trained_Ms[best_index]
    mean = means[best_index]
    std = stds[best_index]

    x_test -= mean
    x_test /= std

    print("Predicting data...")
    s_classify = time.time()
    y_pred = chosen_model.predict(x_test)
    y_pred = np.array(y_pred)
    y_pred = y_pred.ravel()

    e_classify = time.time()
    print("Predicting data...OK, took: " + str((e_classify - s_classify)))

    mse, rsq = get_performance_vals(y_test, y_pred)
    print(list(zip(y_test,y_pred)))

    return chosen_model, mean, std, mse, rsq

def main():

    parameters = [['IVF_simplemodel', 0.1, 0.2], ['IVF_simplemodel', 0.1, 0.3], ['IVF_simplemodel', 0.2, 0.3], ['IVF_simplemodel', 0.2, 0.4], ['IVF_minimalmodel', 0, 0]]

    df_met = pd.read_csv('13_methylation', sep='\t', index_col = 0)
    df_s = pd.read_csv('filtered_successdatratio', sep='\t', index_col = 0)
    df = df_met.join(df_s, how='inner')
    df.dropna(inplace=True)
    df['noeuploid'] = df['Ratio'] == 0

    y = df.loc[:,'noeuploid']
    x = df.drop(['Numsuccess', 'Ratio', 'Numtrials', 'noeuploid'], axis=1)

    #sss = StratifiedShuffleSplit(5, test_size=0.75, random_state=0)
    sss = StratifiedKFold(n_splits=5, shuffle=True)

    mses = []
    rsqs = []

    train_indices= []
    test_indices= []

    for train_i, test_i in sss.split(x,y):
        train_indices.append(train_i)
        test_indices.append(test_i)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for chosen_model, mean, std, mse, rsq in executor.map(crossvalfunc, repeat(x), repeat(y), repeat(parameters), train_indices, test_indices):
            mses.append(mse)
            rsqs.append(rsq)

    print("Mean squared error is: %.3f (+/- %.3f)" % (np.mean(mses), np.std(mses)))
    print("R squared is %.3f (+/- %.3f)" % (np.mean(rsqs), np.std(rsqs)))

    # serialize model to JSON

    std_name = 'filtered_successpredratio_14_std'
    path = "./saved_models/" + std_name + ".pickle"
    output = open(path, 'w+b')
    pickle.dump(std, output)
    output.close()

    mean_name = 'filtered_successpredratio_14_mean'
    path = "./saved_models/" + mean_name + ".pickle"
    output = open(path, 'w+b')
    pickle.dump(mean, output)
    output.close()

    model_json = chosen_model.to_json()
    model_name = 'filtered_successpredratio_14_model'
    with open("./saved_models/" + model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    chosen_model.save_weights("./saved_models/" + model_name + ".h5")
    print("Saved model to disk")

if __name__ == '__main__':
    main()
