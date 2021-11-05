import numpy as np
from collections import defaultdict

# train dataset -> features
X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]])
# train dataset -> labels
Y_train = ['Y', 'N', 'Y', 'Y']

# test dataset
X_test = np.array([[1, 1, 0]])


def getLabelIndices(labels):
    labelIndices = defaultdict(list)
    for index, label in enumerate(labels):
        labelIndices[label].append(index)
    return labelIndices


labelIndices = getLabelIndices(Y_train)


def getPrior(labelIndices):
    prior = {label: len(indices) for label, indices in
             labelIndices.items()}
    totalCount = sum(prior.values())
    for label in prior:
        prior[label] /= totalCount
    return prior


prior = getPrior(labelIndices)


def getLikelihood(features, labelIndices, smoothing=0):
    likelihood = {}
    for label, indices, in labelIndices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        totalCount = len(indices)
        likelihood[label] = likelihood[label] / (totalCount + 2 * smoothing)
    return likelihood


smoothing = 1
likelihood = getLikelihood(X_train, labelIndices, smoothing)


def getPosterior(X, prior, likelihood):
    posteriors = []
    for x in X:
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                if bool_value:
                    posterior[label] *= likelihood_label[index]
                else:
                    posterior[label] *= (1 - likelihood_label[index])
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


posterior = getPosterior(X_test, prior, likelihood)


print("Label indices :\n", labelIndices, "\n")
print("Prior: \n", prior, "\n")
print('Likelihood:\n', likelihood, "\n")
print('Posterior:\n', posterior)


