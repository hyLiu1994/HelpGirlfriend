import numpy as np
import pandas as pd
import argparse
import os
import json
import dataio
from sklearn.model_selection import KFold
from sklearn import metrics


def getMetric(y_true, y_prob):
    metrics = ['MAE', 'MSE', 'Accuracy', 'Precision', 'AP', 'Recall', 'F1-score', 'AUC']
    classifiers = {'MAE':metrics.mean_absolute_error,
                   'MSE':metrics.mean_squared_error,
                   'Accuracy':metrics.accuracy_score,
                   'Precision':metrics.precision_score,
                   'AP':metrics.average_precision_score,
                   'Recall':metrics.recall_score,
                   'F1-score':metrics.f1_score,
                   'AUC':metrics.roc_auc_score
    }
    results={}

    for classifier in classifiers:
        print ('******************* %s ********************' % classifier)
        #start_time = time.time()
        results[classifier] = classifiers[classifier](y_true, y_prob)
        #print 'training took %fs!' % (time.time() - start_time)
    return results
