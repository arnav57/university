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

    # drop columns with NA
    df.dropna(['Id'], axis=1, inplace=True)

    # drop provided columns
    if dropcols is not None:
        df.drop(dropcols, axis=1, inplace=True)
    
    return df

def one_hot_encode(dataframe, catcols):
    '''one hot encodes all provided categorical colums in catcols = [ ], adding them to the df and keeping track of added/removed column counts'''

    # copy the original df as to not modify it
    df = dataframe.copy(deep=True)

    rm_cols = 0
    add_cols = 0

    for col in catcols:
        # obtain one hot encoding for col
        one_hot = pd.get_dummies(df[col])

        # remove the col from the dataframe
        df.drop([col], axis=1, inplace=True)
        rm_cols = rm_cols + 1

        # generate unique col names (ensure no column overriding), add columnwise to the df
        col_names = list(one_hot.columns)
        add_cols = add_cols + len(add_cols)
        for col_name in col_names:
            name = col_name + "-" + col
            df[name] = one_hot[col_name]
        
        print(f'Added {add_cols} columns, removed {rm_cols}, resulting in a net-gain of {add_cols-rm_cols} columns to the returned dataframe...')

        return df

def binarize(y_target):
    '''converts the continuous valued (listlike) y_target to a binary variable such that 1 >= mean, 0 < mean'''
    y_targ = np.copy(y_target)

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
## MODEL TRAINING FUNCTIONS
## -------------------------------

def train_sklearn_model(x_train, y_train, model=None):
    if model is None:
        raise Exception('No sklearn-model specified to train')
    
    model.fit(x_train, y_train)

    return model

def train_pytorch_model(x_train, y_train, epochs=100, model=None, epsilon=None, lossfcn=None):
    if ((model is None) or (opt is None) or (lossfcn is None)):
        raise Exception('improper torch-model, opt or lossfcn specified')
    
    # convert inputs to torch tensor
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    # define the opt
    opt  = torch.optim.SGD(model.parameters(), lr=epsilon)

    # list to hold losses
    losses = []

    # start training
    for i in range(epochs):

        # reset grad
        opt.zero_grad()

        # obtain current preds
        y_preds = model(x_train)

        # find losses / append to list
        l = lossfcn(input=y_preds, target=y_train)
        losses.append(l.item())

        # calc gradients
        l.backward()

        # train
        opt.step()
    
    return model, losses
    


