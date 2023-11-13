import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

w1, w2 = 0.5, 0.5
m01, m02, m1 = np.array([5, 0]), np.array([0, 4]), np.array([3, 2])
C01, C02, C1 = np.array([[4, 0], [0, 2]]), np.array([[1, 0], [0, 3]]), np.array([[2, 0], [0, 2]])
P_L0, P_L1 = 0.6, 0.4

np.random.seed(0)


def generate_dataset(n, class_prob):
    X = []
    Y = []
    for _ in range(n):
        if np.random.rand() < class_prob:
            # Class 0
            if np.random.rand() < w1:
                x = np.random.multivariate_normal(m01, C01)
            else:
                x = np.random.multivariate_normal(m02, C02)
            y = 0
        else:
            # Class 1
            x = np.random.multivariate_normal(m1, C1)
            y = 1
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


# Datasets
D100_train, D100_train_label = generate_dataset(100, P_L0)
D1000_train, D1000_train_label = generate_dataset(1000, P_L0)
D10000_train, D10000_train_label = generate_dataset(10000, P_L0)
D20K_validate, D20K_validate_label = generate_dataset(20000, P_L0)


def theoretical_classifier_prob(x):
    p_x_L0 = w1 * multivariate_normal.pdf(x, m01, C01) + w2 * multivariate_normal.pdf(x, m02, C02)
    p_x_L1 = multivariate_normal.pdf(x, m1, C1)
    return p_x_L1 * P_L1 / (p_x_L1 * P_L1 + p_x_L0 * P_L0)


X_val, Y_val = D20K_validate, D20K_validate_label
prob_scores = [theoretical_classifier_prob(x) for x in X_val]

# ROC curve
fpr, tpr, _ = roc_curve(Y_val, prob_scores)

# Plotting ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.scatter(fpr[np.argmin(np.abs(tpr - (1 - fpr)))], tpr[np.argmin(np.abs(tpr - (1 - fpr)))], marker='o', color='red',
            label='Min-P(error)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

predicted_labels = [1 if score > 0.5 else 0 for score in prob_scores]

min_P_error = np.mean(np.array(predicted_labels) != Y_val)
print(min_P_error)


# Part 2

def estimations_part2(dataset_x, dataset_labels):
    x_L0 = dataset_x[dataset_labels == 0]
    x_L1 = dataset_x[dataset_labels == 1]
    n_L0 = x_L0.shape[0]
    n_L1 = x_L1.shape[0]
    n_tot = n_L0 + n_L1
    est_P_L0 = n_L0 / n_tot
    est_P_L1 = n_L1 / n_tot

    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(x_L0)
    weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_
    mean_L1 = np.mean(x_L1, axis=0)
    cov_L1 = np.cov(x_L1, rowvar=False)
    return est_P_L0, est_P_L1, weights, means, covariances, mean_L1, cov_L1


def classify(x, est_P_L0, est_P_L1, weights, means, covariances, mean_L1, cov_L1):
    # L = 0
    prob_L0 = sum([w * multivariate_normal(mean, cov).pdf(x) for w, mean, cov in zip(weights, means, covariances)])
    prob_L0 *= est_P_L0

    # L = 1
    prob_L1 = multivariate_normal(mean_L1, cov_L1).pdf(x)
    prob_L1 *= est_P_L1

    return prob_L1 / (prob_L0 + prob_L1)


# generate estimated params:
est_P_L0, est_P_L1, weights, means, covariances, mean_L1, cov_L1 = estimations_part2(D10000_train, D10000_train_label)
Best_P_L0, Best_P_L1, Bweights, Bmeans, Bcovariances, Bmean_L1, Bcov_L1 = estimations_part2(D1000_train, D1000_train_label)
Cest_P_L0, Cest_P_L1, Cweights, Cmeans, Ccovariances, Cmean_L1, Ccov_L1 = estimations_part2(D100_train, D100_train_label)


predictions = np.array([classify(x, est_P_L0, est_P_L1, weights, means, covariances, mean_L1, cov_L1) for x in D20K_validate])
Bpredictions = np.array([classify(x, Best_P_L0, Best_P_L1, Bweights, Bmeans, Bcovariances, Bmean_L1, Bcov_L1) for x in D20K_validate])
Cpredictions = np.array([classify(x, Cest_P_L0, Cest_P_L1, Cweights, Cmeans, Ccovariances, Cmean_L1, Ccov_L1) for x in D20K_validate])


# Calculate ROC curve
fpr_10k, tpr_10k, _ = roc_curve(D20K_validate_label, predictions)
fpr_1k, tpr_1k, _ = roc_curve(D20K_validate_label, Bpredictions)
fpr_100, tpr_100, _ = roc_curve(D20K_validate_label, Cpredictions)


# Plot ROC curve
plt.figure()
plt.plot(fpr_10k, tpr_10k, label='ROC curve - 10k_train')
plt.plot(fpr_1k, tpr_1k, label='ROC curve - 1k_train')
plt.plot(fpr_100, tpr_100, label='ROC curve - 100_train')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

predicted_labels_p2 = [1 if score > 0.5 else 0 for score in predictions]
Bpredicted_labels_p2 = [1 if score > 0.5 else 0 for score in Bpredictions]
Cpredicted_labels_p2 = [1 if score > 0.5 else 0 for score in Cpredictions]


# Calculate minimum probability of err
min_P_error = np.mean(np.array(predicted_labels_p2) != Y_val)
Bmin_P_error = np.mean(np.array(Bpredicted_labels_p2) != Y_val)
Cmin_P_error = np.mean(np.array(Cpredicted_labels_p2) != Y_val)

print(min_P_error, Bmin_P_error, Cmin_P_error)


# Part 3
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def negative_log_likelihood(beta, X, y):
    logits = np.dot(X, beta)
    log_likelihood = -np.sum(y * np.log(sigmoid(logits)) + (1 - y) * np.log(1 - sigmoid(logits)))
    return log_likelihood


def train_logistic_regression(X_train, y_train):
    initial_beta = np.zeros(X_train.shape[1])
    result = minimize(negative_log_likelihood, initial_beta, args=(X_train, y_train))
    return result.x


def predict(X, beta):
    logits = np.dot(X, beta)
    return sigmoid(logits) >= 0.5


beta_10000 = train_logistic_regression(D10000_train, D10000_train_label)
predictions_10000 = predict(X_val, beta_10000)

beta_1000 = train_logistic_regression(D1000_train, D1000_train_label)
predictions_1000 = predict(X_val, beta_1000)

beta_100 = train_logistic_regression(D100_train, D100_train_label)
predictions_100 = predict(X_val, beta_100)


# Estimate probability of error for each model
error_10000 = 1 - accuracy_score(Y_val, predictions_10000)
error_1000 = 1 - accuracy_score(Y_val, predictions_1000)
error_100 = 1 - accuracy_score(Y_val, predictions_100)

print(error_10000, error_1000, error_100)

def train_logistic_regression_2(X_train, y_train):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)

    initial_beta = np.zeros(X_train_poly.shape[1])
    result = minimize(negative_log_likelihood, initial_beta, args=(X_train_poly, y_train))
    return result.x, poly


def predict_2(X, beta, poly):
    X_poly = poly.transform(X)
    logits = np.dot(X_poly, beta)
    return sigmoid(logits) >= 0.5


beta_10000, poly_10000 = train_logistic_regression_2(D10000_train, D10000_train_label)
predictions_10000 = predict_2(X_val, beta_10000, poly_10000)
beta_1000, poly_1000 = train_logistic_regression_2(D1000_train, D1000_train_label)
predictions_1000 = predict_2(X_val, beta_1000, poly_1000)
beta_100, poly_100 = train_logistic_regression_2(D100_train, D100_train_label)
predictions_100 = predict_2(X_val, beta_100, poly_100)



error_10000 = 1 - accuracy_score(Y_val, predictions_10000)
error_1000 = 1 - accuracy_score(Y_val, predictions_1000)
error_100 = 1 - accuracy_score(Y_val, predictions_100)
print(error_10000, error_1000, error_100)











