import numpy as np
from sklearn.naive_bayes import BernoulliNB


X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]])
Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[1, 1, 0]])

# using the BernoulliNB module
# smoothing factor is alpha=1.0
clf = BernoulliNB(alpha=1.0, fit_prior=True)
# train the Naive Bayes classifier with the fit method
clf.fit(X_train, Y_train)

# predicted probability
pred_prob = clf.predict_proba(X_test)
pred = clf.predict(X_test)

print('[scikit-learn] Predicted probabilities:\n', pred_prob, "\n")
print('[scikit-learn] Prediction:', pred)
