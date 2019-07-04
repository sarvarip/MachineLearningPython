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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import IVF_simplemodel as trainer
import random
import time
import pickle
from keras.optimizers import Adam
from keras.models import model_from_json
from imblearn.over_sampling import SMOTE
from collections import Counter
import math

df_met = pd.read_csv('13_methylation', sep='\t', index_col = 0)
df_s = pd.read_csv('filtered_successdatratio', sep='\t', index_col = 0)
df = df_met.join(df_s, how='inner')
df.dropna(inplace=True)
df['noeuploid'] = df['Ratio'] == 0

y = df.loc[:,'noeuploid']
x = df.drop(['Numsuccess', 'Ratio', 'Numtrials', 'noeuploid'], axis=1)

def get_performance_vals(y_test, classes):
    a = np.array(y_test)
    b = classes
    print("Predicted and actual classes\n")
    print(classes)
    print(a)
    tp = np.sum(np.multiply(a==1, b==1)) #TP
    fp = np.sum(np.multiply(b==1, a==0)) #FP
    tn = np.sum(np.multiply(a==0, b==0)) #TN
    fn = np.sum(np.multiply(a==1, b==0)) #FN

    tp = int(tp)
    fp = int(fp)
    tn = int(tn)
    fn = int(fn)

    print("True positive: %d, false positive: %d, true negative: %d, false negative: %d\n" %(tp,fp,tn,fn))

    if tp==0:
        fscore = 0
        if fp==0:
            mcc = 0
        else:
            mcc = (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    else:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        fscore = 2*precision*recall/(precision+recall)
        mcc = (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))


    precision = tp/(tp+fp)
    specificity = tn/(tn+fp)
    recall = tp/(tp+fn)
    accuracy = np.sum(a==b)/len(classes)

    print("%s: %2f" % ('MCC', mcc))
    print("%s: %.2f%%" % ('Accuracy', 100*accuracy))
    print("%s: %.2f" % ('F1 score', fscore))
    print("%s: %.2f" % ('Precision', precision))
    print("%s: %.2f" % ('Recall', recall))
    print("%s: %.2f" % ('Specificity', specificity))

    return mcc, accuracy, fscore, precision, recall, specificity


def do_kfolds(x, y):
    kfold = StratifiedKFold(n_splits=4, shuffle=True)
    MCCs = []
    Accuracies = []
    F1s = []
    Precisions = []
    Recalls = []
    Specificities = []

    x = np.array(x)
    y = np.array(y)
    print("Preparing training and label data...OK")
    print("")

    for train, test in kfold.split(x, y):

        x_train = x[train]
        y_train = y[train]
        x_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
        print(sorted(Counter(y_resampled).items()))
        x_train = x_resampled
        y_train = y_resampled
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
        M = trainer.build_sequential_model(0.2,0.3,shape = x_train.shape[1])
        trained_M = trainer.fit_model_batch(M, x_train, y_train, num_epoch=2000)

        print("Predicting data...")
        s_classify = time.time()
        scores = trained_M.predict(x_test)
        classes = trained_M.predict_classes(x_test)
        classes = np.array(classes)
        classes = classes.ravel()

        e_classify = time.time()
        print("Predicting data...OK, took: " + str((e_classify - s_classify)))

        mcc, accuracy, fscore, precision, recall, specificity = get_performance_vals(y_test, classes)

        MCCs.append(mcc)
        Accuracies.append(accuracy)
        F1s.append(fscore)
        Precisions.append(precision)
        Recalls.append(recall)
        Specificities.append(specificity)

        print(list(zip(y_test,scores)))

    print("MCC is: %.3f (+/- %.3f)" % (np.mean(MCCs), np.std(MCCs)))
    print("Accuracy is %.3f (+/- %.3f)" % (np.mean(Accuracies), np.std(Accuracies)))
    print("F1 score is %.3f (+/- %.3f)" % (np.mean(F1s), np.std(F1s)))
    print("Precision is %.3f (+/- %.3f)" % (np.mean(Precisions), np.std(Precisions)))
    print("Recall is %.3f (+/- %.3f)" % (np.mean(Recalls), np.std(Recalls)))
    print("Specificity is %.3f (+/- %.3f)" % (np.mean(Specificities), np.std(Specificities)))

    return trained_M, mean, std

trained_M, mean, std = do_kfolds(x,y)

# serialize model to JSON

std_name = 'class_successpredratio_best1_std'
path = "./saved_models/" + std_name + ".pickle"
output = open(path, 'w+b')
pickle.dump(std, output)
output.close()

mean_name = 'class_successpredratio_best1_mean'
path = "./saved_models/" + mean_name + ".pickle"
output = open(path, 'w+b')
pickle.dump(mean, output)
output.close()

model_json = trained_M.to_json()
model_name = 'class_successpredratio_best1_model'
with open("./saved_models/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
trained_M.save_weights("./saved_models/" + model_name + ".h5")
print("Saved model to disk")
