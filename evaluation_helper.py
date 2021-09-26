import numpy as np
import pandas as pd
import logging

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

import os.path
import json

class evaluation_helper:
    def __init__(self, filename='./data/results.json'):
        self.filename = filename
        if os.path.isfile(filename):
            with open(filename) as f:
              self.results = json.load(f)
        else:
            self.results = {}

    def evaluate(self, name, y_test, y_pred, time):
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f'Completed in {time:.2f}s')
        print('confusion matrix: \n', confusion_matrix(y_test, y_pred), '\n')
        print('f1= ', f1, '; accuracy= ', accuracy, '; precision= ', precision, '; recall= ', recall)
        print('roc_auc= ', roc_auc)
        
        self.results[name] = {}
        self.results[name]['f1'] = f1
        self.results[name]['accuracy'] = accuracy
        self.results[name]['precision'] = precision
        self.results[name]['recall'] = recall
        self.results[name]['roc_auc'] = roc_auc
        self.results[name]['time'] = time
        
        self.save()
        return f1, accuracy, precision, recall, roc_auc


    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.results, f)
            
    def to_df(self):
        return pd.DataFrame.from_dict(self.results, orient='index');
