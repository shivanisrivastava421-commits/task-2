import numpy as np 
import math
# Setting a global epsilon for use in log probabilities to avoid math domain errors
EPSILON = 1e-9

class NaiveBayes:
    def fit(self, X, y):
        self.classes = set(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            num_samples_c = len(X_c)
            if num_samples_c == 0:
                self.priors[c] = EPSILON  # Assign a small prior if no samples in class
                self.mean[c] = np.zeros(X.shape[1])
                self.var[c] = np.ones(X.shape[1]) * EPSILON
            else:
                self.mean[c] = X_c.mean(axis=0)
                # Add small value to variance to avoid division by zero and log(0)
                self.var[c] = X_c.var(axis=0) + EPSILON  
                self.priors[c] = num_samples_c / len(X)

    def gaussian_prob(self, class_idx, x, c):
        """
        Calculates the Gaussian probability (PDF) for a given feature value.
        NOTE: The variable definitions must come BEFORE the return statement.
        """
        mean = self.mean[c][class_idx]
        var = self.var[c][class_idx]
        
        # 1. Define the components (numerator and denominator)
        numerator = math.exp(-((x - mean) ** 2) / (2 * var))
        denominator = math.sqrt(2 * math.pi * var)
        
        # 2. Return the calculated probability (plus epsilon to prevent log(0))
        return (numerator / denominator) + EPSILON

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                # Use log probabilities to avoid underflow
                prior = math.log(self.priors[c])
                
                # Sum the log of conditional probabilities (log(P(x|c)))
                conditional = sum(math.log(self.gaussian_prob(i, x[i], c)) for i in range(len(x)))
                
                posteriors[c] = prior + conditional
                
            y_pred.append(max(posteriors, key=posteriors.get))
        return y_pred
