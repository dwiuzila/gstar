import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings


class GSTAR:
    def __init__(self, z, w, p, lambd):
        self.w = w
        self.p = p
        self.lambd = lambd
        self.col = z.columns
        self.idx = z.index
        self.z = z.values
        self.weights = [np.eye(z.shape[1])] + self.w
        self._check()

    def _check(self):
        if not (isinstance(self.w, list) and isinstance(self.lambd, list)):
            raise TypeError("Both weight matrices and lambda should be a list object.")
            
        if not isinstance(self.p, int) or self.p < 1:
            raise ValueError("The time order p must be positive integers.")
        
        for l in self.lambd:
            if not isinstance(l, int) or l < 0:
                raise ValueError("The space order lambda must be nonnegative integers.")
        
        if self.p != len(self.lambd):
            raise ValueError("The number of lambda (%d) doesn't match the time order p (%d)."
                             % (len(self.lambd), self.p))
        
        if max(self.lambd) > len(self.w):
            raise ValueError("The number of weight matrices (%d) is insufficient for maximum lambda (%d)."
                             % (len(self.w), max(self.lambd)))
        elif max(self.lambd) < len(self.w):
            warnings.warn("Too many weight matrices (%d) compared to maximum lambda (%d). "
                          "Some weight matrices may not be used."
                          % (len(self.w), max(self.lambd)))
        
        n = self.z.shape[1]
        for weight in self.w:
            if np.sum(np.isnan(weight)) > 0:
                raise ValueError("Weight contains NaN.")
            
            i,j = weight.shape
            if i != n or j != n:
                raise ValueError("Dimension mismatch. "
                                 "Weight should have %d by %d dimension."
                                 % (n,n))

    @staticmethod
    def pad_zero(x):
        result = None
        for row in x:
            if result is None:
                result = np.diag(row)
            else:
                result = np.concatenate((result, np.diag(row)))
        return np.array(result)
    
    def fit(self):
        # create Y
        Y = self.z[self.p:].ravel()

        # create X
        m,n = self.z.shape

        wz = []
        for weight in self.weights:
            wz.append(self.z @ weight.T)

        xcol = []
        for k, s_lag in enumerate(self.lambd):
            for l in range(s_lag+1):
                wzcol = wz[l][self.p-k-1:-k-1]
                wzcol = self.pad_zero(wzcol)
                xcol.append(wzcol)

        X = np.concatenate(xcol, axis=1)

        # solve params
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X, Y)
        z_hat = lr.predict(X)
        z_hat = z_hat.reshape(m - self.p, n)
        self.params = lr.coef_
        self.fitted = pd.DataFrame(z_hat, columns=self.col, index=self.idx[self.p:])

    def predict(self, forecast_time):
        z_pred = self.z[-self.p:]
        preds = []

        for _ in range(forecast_time):
            wz = []
            for weight in self.weights:
                wz.append(z_pred @ weight.T)
            
            xcol = []
            for k, s_lag in enumerate(self.lambd):
                for l in range(s_lag+1):
                    wzcol = np.atleast_2d(wz[l][-k-1])
                    wzcol = self.pad_zero(wzcol)
                    xcol.append(wzcol)
                    
            X = np.concatenate(xcol, axis=1)
            pred = X @ self.params
            preds.append(pred)
            z_pred = np.concatenate((z_pred[1:], np.atleast_2d(pred)))

        preds = np.array(preds)
        preds = pd.DataFrame(preds, columns=self.col)
        return preds