from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, roc_auc_score
from IrisMatching import DISTANCE_METRICS, LinearDiscriminantAnalysis, iris_matching
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_crr(y_preds, y_true):
    return [(y_pred == y_true).mean() for y_pred in y_preds]

def generate_crr_table(crrs, metrics=DISTANCE_METRICS):
    for i, metric in enumerate(metrics):
        print(f"With {metric} metric, CRR is {crrs[i]}")

def generate_LDA_dimension_CRR_plot(x_train, y_train, x_test, y_test, 
                                    dimension_arr = list(range(10, 107, 10)) + [107]):
    crr_arr = []
    for i in dimension_arr:
        lda = LinearDiscriminantAnalysis(n_components = i).fit(x_train, y_train)
        x_train_lda = lda.transform(x_train)
        x_test_lda = lda.transform(x_test)
        clfs, y_preds = iris_matching(x_train_lda, y_train, x_test_lda)
        crr_cosine = calc_crr(y_preds, y_test)[2]
        crr_arr.append(crr_cosine)
     
    plt.figure()
    plt.plot(dimension_arr, crr_arr)
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct Recognition rate')
    plt.title('Recognition results using feature of different vector')
    #plt.show()
    return crr_arr
        
# use distance to generate ROC curve
def predict_proba(self, X):
    distances = pairwise_distances(X, self.centroids_, metric=self.metric)
    probs = np.min(distances, axis=1)
    return probs


def generate_roc_curve(clfs, y_preds, x_test, y_true,
                       metrics=DISTANCE_METRICS):
    plt.figure()
    lw = 2

    for i, metric in enumerate(metrics):
        score = predict_proba(clfs[i], x_test)
        fpr, tpr, _ = roc_curve(y_true != y_preds[i], score)
        roc_auc = roc_auc_score(y_true != y_preds[i], score)
        plt.plot(fpr,
                 tpr,
                 lw=lw,
                 label=f'ROC curve for {metric} (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for verification phase')
    plt.legend(loc="lower right")
    #plt.show()

def calc_fm_fnm(clf, y_pred, x_test, y_true, threshold):
    # use the cosine distance  
    score = predict_proba(clf, x_test) #pred_dist
    
    correct_pred_idx = np.where(y_pred == y_true)[0]
    wrong_pred_idx = np.where(y_pred != y_true)[0]
    
    # false match given the threshold: dist<t but y_pred != y_test
    fm = np.sum(score[wrong_pred_idx]<threshold)
    # false non-match: dist>t but y_pred = y_test
    fnm = np.sum(score[wrong_pred_idx]>=threshold)
    TN = np.sum(score[correct_pred_idx]>=threshold)
    TP = np.sum(score[correct_pred_idx]<threshold)
    # FP / (FP + TN)
    fmr = fm/(fm+TN)
    # FN / (FN + TP)
    fnmr = fnm/(fnm+TP)
    
    return fmr, fnmr

def generate_fmr_fnmr_arr(clf, y_pred, x_test, y_true, thresholds):
    fmr_arr = []
    fnmr_arr = []
    for t in thresholds:
        fmr, fnmr = calc_fm_fnm(clf, y_pred, x_test, y_true, t)
        fmr_arr.append(fmr)
        fnmr_arr.append(fnmr)
    
    return fmr_arr, fnmr_arr

def generate_threshold_table(clf, y_pred, x_test, y_true, thresholds):
    fmr_arr, fnmr_arr = generate_fmr_fnmr_arr(clf, y_pred, x_test, 
                                              y_true, thresholds)
    print('========= FMR and FNMR table =========')
    value_dict = {'Threshold': thresholds,
                'False match rate': fmr_arr,
                 'False non-match rate': fnmr_arr}
    print(pd.DataFrame(value_dict))
    
def generate_fm_fnm_curve(clf, y_pred, x_test, y_true, thresholds):
    fmr_arr, fnmr_arr = generate_fmr_fnmr_arr(clf, y_pred, x_test, 
                                              y_true, thresholds)
    plt.figure()
    plt.plot(fmr_arr, fnmr_arr)
    plt.xlabel('False match rate')
    plt.ylabel('False non-match rate')
    plt.title('FMR vs FNMR')
    plt.show()