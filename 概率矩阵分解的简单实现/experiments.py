import numpy as np
import pandas as pd
import argparse
import os
import json
import dataio
from sklearn.model_selection import KFold
from sklearn import metrics
from getMetrics import getMetric
import MF

def transferArray2List(I,R):
    """Preprocess ASSISTments 2012-2013 dataset.

    Arguments:
    I -- the interaction records (ndarray)
    R -- the results records (ndarray)

    Outputs:
    y -- the results records (list)
    """
    y=[]
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if I[i][j]==1:
                y.append(R[i][j])
    return y

if __name__ == "__main__":
    results={}
    R_true=np.load(os.path.join('proData', options.dataset,'R_mat.npy'))
    CURRENT_READ_FOLDER=os.path.join(READ_FOLDER, str(run_id))
    CURRENT_SAVE_FOLDER=dataio.build_new_paths(SAVE_FOLDER, str(run_id))
    I_train=np.load(os.path.join(CURRENT_READ_FOLDER,'I_train.npy'))
    I_test=np.load(os.path.join(CURRENT_READ_FOLDER,'I_test.npy'))
    model = MF(max_iter=400)
    model.fit(I_train, R_true)
    R_prob=model.predict(I_test, R_true)
    y_true=transferArray2List(I_test, R_true)
    y_prob=transferArray2List(I_test, R_prob)
    results=getMetric(y_true, y_prob)
    dataio.SaveDict(results,SAVE_FOLDER,'results.json')
