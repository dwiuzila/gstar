"""Available models"""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag


class GSTAR:
    def __init__(self, weights, p, lambd):
        self.weights = weights
        self.p = p
        self.lambd = lambd
        self.result = None
        self.sanity_check()
    
    def sanity_check(self):
        if not (isinstance(self.weights, list) and isinstance(self.lambd, list)):
            raise TypeError("Both weight matrices and lambda should be a list object.")
            
        if not isinstance(self.p, int) or self.p < 1:
            raise ValueError("The time order p must be positive integers.")
        
        for l in self.lambd:
            if not isinstance(l, int) or l < 0:
                raise ValueError("The spatial order lambda must be nonnegative integers.")
        
        if self.p != len(self.lambd):
            raise ValueError("The number of lambda (%d) doesn't match the time order p (%d)."
                             % (len(self.lambd), self.p))
        
        if max(self.lambd) + 1 > len(self.weights):
            raise ValueError("The number of weight matrices (%d) is insufficient for maximum lambda (%d)."
                             % (len(self.weights), max(self.lambd)))
        
        for weight in self.weights:
            if np.sum(np.isnan(weight)) > 0:
                raise ValueError("Weight contains NaN.")
    
    def fit(self, train):
        # initialization
        if isinstance(train, pd.DataFrame):
            self.idx = train.index
            self.col = train.columns
        elif isinstance(train, np.ndarray):
            self.idx = range(train.shape[0])
            self.col = range(train.shape[1])
        else:
            raise TypeError("Train dataset must be either numpy array or pandas dataframe.")
        
        self.train = np.asarray(train)
        
        # sanity check
        self.n = len(self.col)
        for weight in self.weights:
            i,j = weight.shape
            if i != self.n or j != self.n:
                raise ValueError("Dimension mismatch. "
                                 "Weight should have %d by %d dimension."
                                 % (self.n, self.n))
        
        # create X
        Vs = self.intermediary_matrix(self.train)
        X = self.preprocess_data(Vs)

        # create Y
        Y = self.train[self.p:,:]
        Y_copy = Y.copy()
        Y = Y.reshape(-1, 1, order='F')

        # create B
        B = np.linalg.lstsq(X, Y, rcond=-1)[0]

        # check stationarity status
        dets, stationary = self.stationarity_check(B, self.p, self.lambd)

        # fit the model
        Y_hat = X @ B
        msr, aic = self.score(Y, Y_hat, B)
        Y_hat = Y_hat.reshape(-1, self.n, order='F')
        residual = Y_copy - Y_hat
        fitted = pd.DataFrame(Y_hat, columns=self.col, index=self.idx[self.p:])
        
        # tabulate result
        self.result = {
            'p': self.p,
            'lambda': self.lambd,
            'MSR': msr,
            'AIC': aic,
            'determinants': dets,
            'stationary': stationary,
            'residual': residual,
            'parameter': B
        }

        return fitted
    
    def intermediary_matrix(self, data, mode='train'):
        Vs = []
        for k in range(self.p):
            for el in range(self.lambd[k] + 1):
                V = self.weights[el] @ data.T
                if mode == 'train':
                    V = V[:, self.p-k-1:-k-1]
                elif mode == 'test':
                    V = V[:, -k-1]
                Vs.append(V.reshape(self.n, -1))
        Vs = np.hstack(Vs)
        return Vs
    
    def preprocess_data(self, Vs):
        X = None
        for row in Vs:
            temp = row.reshape(-1, self.p + sum(self.lambd), order='F')
            if X is None:
                X = block_diag(temp)
            else:
                X = block_diag(X, temp)
        return X
    
    def stationarity_check(self, B, p, lambd):
        # generate phi
        B = B.reshape(-1, self.n, order='F')
        phi = []
        for row in B:
            phi.append(np.diag(row))

        # create A
        As = [np.eye(self.n)]
        i = 0
        for k in range(p):
            A = 0
            for el in range(lambd[k] + 1):
                A += phi[i] @ self.weights[el]
                i += 1
            As.append(A)

        # calculate M
        M = []
        for i in range(1, p+1):
            Mrow = []
            for j in range(1, p+1):
                if i == j:
                    Mij = 0
                    for k in range(i):
                        Mij += As[k].T @ As[k] - As[p-k].T @ As[p-k]
                elif j < i:
                    Mij = - As[i-j] - As[p-i+j].T @ As[p]
                    for k in range(1, j):
                        Mij += As[k].T @ As[k+i-j] - As[p-k-i+j].T @ As[p-k]
                else:
                    Mij = - As[j-i] - As[p-j+i].T @ As[p]
                    for k in range(1, i):
                        Mij += As[k].T @ As[k+j-i] - As[p-k-j+i].T @ As[p-k]
                    Mij = Mij.T

                Mrow.append(Mij)
            M.append(Mrow)
        M = np.block(M)

        # check stationarity status and determinants
        stationary = True
        dets = []
        for i in range(p * self.n):
            det = np.linalg.det(M[:i+1, :i+1])
            if det < 0: stationary = False
            dets.append(det)

        return dets, stationary
    
    def score(self, y_true, y_pred, B):
        T, N = len(y_true.ravel()), len(B)
        res = y_true.ravel() - y_pred.ravel()
        tot_error = res.dot(res)
        msr = tot_error / T
        log_likelihood = - T/2 * np.log(2*np.pi) - T/2 * np.log(msr) - 1/(2*msr) * tot_error
        AIC = - 2/T * log_likelihood + 2/T * N
        return msr, AIC
    
    def forecast(self, num_iter, test=None):
        if num_iter < 0:
            raise ValueError("The number of forecast time must be positive.")
        
        temp = self.train[-self.p:, :].copy()
        fct = np.vstack((temp, np.zeros((num_iter, self.n))))
        
        if test is not None and num_iter > len(test):
            raise ValueError("The number of forecast time must be no more than time in test dataset.")
        elif test is not None:
            self.test = np.vstack((temp, np.asarray(test)))

        for i in range(num_iter):
            if test is None:
                Vs = self.intermediary_matrix(fct[i:i+self.p, :], mode='test')
            else:
                Vs = self.intermediary_matrix(self.test[i:i+self.p, :], mode='test')
            
            X = self.preprocess_data(Vs)
            Y_hat = X @ self.result['parameter']
            fct[i+self.p] = Y_hat.ravel()

        fct = fct[self.p:, :]
        fct = pd.DataFrame(fct)
        if isinstance(test, pd.DataFrame):
            fct.index = test.index[:num_iter]
            fct.columns = test.columns
        
        return fct