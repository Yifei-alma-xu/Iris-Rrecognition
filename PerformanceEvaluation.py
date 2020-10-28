from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, roc_auc_score
from IrisMatching import DISTANCE_METRICS
import numpy as np
import matplotlib.pyplot as plt


def calc_crr(y_preds, y_true):
    return [(y_pred == y_true).mean() for y_pred in y_preds]

def generate_crr_table(crrs, metrics=DISTANCE_METRICS):
    for i, metric in enumerate(metrics):
        print(f"With {metric} metric, CRR is {crrs[i]}")

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
        ## pos_label = 1 by default
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
    plt.show()