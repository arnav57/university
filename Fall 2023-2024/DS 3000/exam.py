import pandas as pd
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc

## BEGIN:
## DATA PREPROCESSING FUNCTIONS
## -------------------------------

def preprocessing(dataframe, dropcols=None):
    '''Returns a copy of a dataframe with modifications. Drops rows with NULL values, and drops columns provided as kwarg dropcols=[ ]'''

    # copy the original df as to not modify it
    df = dataframe.copy(deep=True)

    # drop rows with NA
    df.dropna(axis=0, inplace=True)

    # drop provided columns
    if dropcols is not None:
        df.drop(dropcols, axis=1, inplace=True)
    
    return df

def one_hot_encode(dataframe, catcols):
    '''one hot encodes all provided categorical colums in catcols = [ ], adding them to the df and keeping track of added/removed column counts'''

    # copy the original df as to not modify it
    df = dataframe.copy(deep=True)

    add_cols = 0
    rm_cols = 0

    for col in catcols:
        # obtain one hot encoding for col
        one_hot = pd.get_dummies(df[col])

        # remove the col from the dataframe
        df.drop([col], axis=1, inplace=True)
        rm_cols = rm_cols + 1

        # generate unique col names (ensure no column overriding), add columnwise to the df
        col_names = list(one_hot.columns)
        for col_name in col_names:
            add_cols = add_cols + 1
            name = col_name + "-" + col
            df[name] = one_hot[col_name]
        
    print(f'Added {add_cols} columns, removed {rm_cols}, resulting in a net-gain of {add_cols-rm_cols} columns to the returned dataframe...')

    return df

def binarize(y_target, type='mean'):
    '''converts the continuous valued (listlike) y_target to a binary variable such that 1 >= mean, 0 < mean'''
    y_targ = np.copy(y_target)

    if type =='mean':
        mean = np.mean(y_targ)
    elif type == 'median':
        mean = np.median(y_targ)
    else:
        mean = np.mean(y_targ)

    for i in range(len(y_targ)):
        if y_targ[i] >= mean:
            y_targ[i] = 1
        else:
            y_targ[i] = 0
    return y_targ

## -------------------------------
## BEGIN:
## PERFORMANCE METRIC FUNCTIONS
## -------------------------------

def rss_tss_r2(y_true, y_preds, print=False):
    '''calculates and returns RSS, TSS and R^2 values from provided y_true, y_preds. Prints results if print=True'''

    y_mean = np.mean(y_true)

    rss = np.sum( (y_true - y_preds)**2 )
    tss = np.sum( (y_true - np.array(y_mean))**2 )
    r2 = 1 - (rss/tss)

    if print:
        print(f'RSS: {rss:.4f}')
        print(f'TSS: {tss:.4f}')
        print(f'R2: {r2:.4f}')

    return (rss, tss, r2)

def prec_acc_rec_f1(y_true, y_preds, print=False):
    '''calculates and returns Precision, Accuracy, Recall, and F1 scores from provided y_true, y_preds. Prints results if print=True'''
    
    prec = precision_score(y_true, y_preds)
    acc = accuracy_score(y_true, y_preds)
    rec = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)

    if print:
        print(f'Precision Score: {prec:.4f}')
        print(f'Accuracy Score: {acc:.4f}')
        print(f'Recall Score: {rec:.4f}')
        print(f'F1 Score: {f1:.4f}')
        

    return prec, acc, rec, f1

def reciever_operating_characteristic(y_true, y_preds, print=False):
    '''Calcs and returns fpr, tpr, thresholds, auc for the provided y_true, and y_preds'''
    # you can plot the ROC curve with plt.plot(fpr, tpr) 

    fpr, tpr, thresholds = roc_curve(y_true, y_preds)
    auc_val = auc(fpr, tpr)

    if print:
        print(f'FPR: {fpr:.4f}')
        print(f'TPR: {tpr:.4f}')
        print(f'AUC (Area Under Curve): {auc:.4f}')
    
    return fpr, tpr, thresholds, auc_val

## -------------------------------
## BEGIN:
## MODEL FUNCTIONS
## -------------------------------

class NN(torch.nn.Module):
    def __init__(self, feats_in, num_classes, architecture):
        self.feats_in = feats_in # input vector dim
        self.num_classes = num_classes # output vector dim
        super().__init__()

        ## define NN architecture
        self.layers = architecture
    
    def forward(self, X):
        # # might be required if you accept np tensor inputs
        # X = X.to(torch.float32)
        X = self.layers(X)
        
        return X

## -------------------------------
## BEGIN:
## TESTBENCH
## -------------------------------

def test_processing():
  # use property price csv from midterm
  dropcols = ['Id']
  data = preprocessing(df, dropcols=dropcols)
  assert data.shape == (1452,27)

  return data

def test_one_hot_encode():
  catcols = ['LandSlope', 'HouseStyle', 'Heating', 'CentralAir', 'PavedDrive']
  data = one_hot_encode(df, catcols=catcols)
  assert data.shape == (1460,45)

  return data

def test_binarize():
  data = df['1stFlrSF']
  output = binarize(data, type='mean')
  mean = np.mean(data)
  for i, data in enumerate(data):
    if (data >= mean):
      assert 1 == output[i]
    else:
      assert 0 == output[i]

  data = df['1stFlrSF']
  output = binarize(data, type='median')
  median = np.median(data)
  for i, data in enumerate(data):
    if (data >= median):
      assert 1 == output[i]
    else:
      assert 0 == output[i]

  return output

