"""
A script that loads JAK2 ligand and decoys and runs an SVM classifier
"""

from learning import loadjak2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

# find repository root
root = os.path.dirname(os.path.dirname(os.getcwd()))

# load JAK2 data and split them into k-fold cross-validation sets
loader = loadjak2(root)
loader.featurize("fragments")
print(np.sum(loader.labels)/loader.labels.shape[0])

splits = loader.split_and_scale(kfold_shuffle=True, num_splits=10, scale_labels=False)

score = []

for split in splits:
    X_train, X_test, X_scaler, y_train, y_test, y_scaler = split

    classifier = SVC(C=1., kernel="rbf")
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_function(X_test)

    # the JAK2 library has 107 ligands vs. 6500 decoys (1.6 percent),
    # which is why the ROC is so high
    score.append(roc_auc_score(y_test, y_score))

print(np.mean(score))


if __name__ == '__main__':
    print("")