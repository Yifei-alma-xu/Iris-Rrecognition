from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors.nearest_centroid import NearestCentroid
import warnings
warnings.filterwarnings("ignore")

DISTANCE_METRICS = ['l1', 'l2', 'cosine']


# use fisher linear discriminant for dimension reduction
def dimension_reduction(x_train, x_test, y_train):
    lda = LinearDiscriminantAnalysis(solver='eigen',
                                     shrinkage='auto').fit(x_train, y_train)
    x_train_lda = lda.transform(x_train)
    x_test_lda = lda.transform(x_test)
    return x_train_lda, x_test_lda


# use nearest center classifier for classification
def iris_matching(x_train, y_train, x_test, metrics=DISTANCE_METRICS):
    clfs = [NearestCentroid(metric=k).fit(x_train, y_train) for k in metrics]
    y_preds = [clf.predict(x_test) for clf in clfs]

    return clfs, y_preds
