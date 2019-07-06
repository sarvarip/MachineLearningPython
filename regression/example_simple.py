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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import IVF_simplemodel as trainer
import random
import time
import pickle
from keras.optimizers import Adam
from keras.models import model_from_json

df_met = pd.read_csv('13_methylation', sep='\t', index_col = 0)
df_s = pd.read_csv('filtered_successdatratio', sep='\t', index_col = 0)

df = df_met.join(df_s, how='inner')
df.dropna(inplace=True)
y = df.loc[:,'Ratio']
x = df.drop(['Numsuccess', 'Ratio', 'Numtrials'], axis=1)

def do_kfolds(x, y):
    random.seed(70)
    kfold = KFold(n_splits=10, shuffle=True, random_state=70)
    mses = []
    rsqs = []
    varexps = []
    x = np.array(x)
    y = np.array(y)
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
        M = trainer.build_sequential_model(0.2, 0.3, shape = x_train.shape[1])
        trained_M = trainer.fit_model_batch(M, x_train, y_train, num_epoch=2000)

        print("Predicting data...")
        s_classify = time.time()
        #scores = trained_M.(x_test)
        y_pred = trained_M.predict(x_test)
        y_pred = np.array(y_pred)
        y_pred = y_pred.ravel()

        e_classify = time.time()
        print("Predicting data...OK, took: " + str((e_classify - s_classify)))

        mse = mean_squared_error(y_test, y_pred)
        stdev = np.std(y_test)
        var = stdev**2
        total_mse = var
        varexp = (total_mse - mse) / total_mse
        corrval = np.corrcoef(y_test, y_pred)[0][1]
        rsq = corrval**2

        mses.append(mse)
        rsqs.append(rsq)
        varexps.append(varexp)

    print("MSE: %.3f (+/- %.3f)" % (np.mean(mses), np.std(mses)))
    print("Variance explained is %.3f (+/- %.3f)" % (np.mean(varexp), np.std(varexp)))
    print("R squared is %.3f (+/- %.3f)" % (np.mean(rsqs), np.std(rsqs)))

    return trained_M, mean, std

trained_M, mean, std = do_kfolds(x,y)

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

model_json = trained_M.to_json()
model_name = 'filtered_successpredratio_14_model'
with open("./saved_models/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
trained_M.save_weights("./saved_models/" + model_name + ".h5")
print("Saved model to disk")
