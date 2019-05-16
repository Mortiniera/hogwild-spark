import numpy as np
from operator import add
from scipy.sparse import csr_matrix
import math
from time import time

# NOTE : It is possible to use the normal hinge loss :
# L(w) = \lambda*||w||^2 + 1/B * sum max(0, 1 - y*(x dot w))
# By commenting the lines for the sparse hinge loss from HOGWILD!
# and uncommenting the other ones. One should then adapt the hyperparms

class SVM:
    def __init__(self, learning_rate, lambda_reg, batch_frac, dim):
        self.__learning_rate = learning_rate
        self.__lambda_reg = lambda_reg
        self.__batch_frac = batch_frac
        self.__dim = dim
        self.__persistence = 20
        self.__w = np.zeros(dim)

    def getW(self):
        ''' Return the weight vector '''
        return self.__w

    def fit(self, data, validation, max_iter, spark):
        ''' Fit the data using the validation set over max_iter or less if it converges earlier '''
        reached_criterion = False
        early_stopping_window = []
        window_smallest = math.inf
        log = []
        w_bc = spark.sparkContext.broadcast(self.__w)
        start_time = time()
        for i in range(max_iter):
            if not reached_criterion:
                # Compute gradient and train loss
                grad, train_loss = self.step(
                    data.sample(False, self.__batch_frac), w_bc)
                self.__w += self.__learning_rate * grad.toarray().ravel()
                # self.__w -= self.__learning_rate * (grad.toarray().ravel() + self.l2_reg_grad(w_bc))
                w_bc = spark.sparkContext.broadcast(self.__w)
                # Compute validation loss and accuracy
                validation_loss = self.loss(validation, w_bc=w_bc)
                # validation_accuracy = self.predict(validation, w_bc=w_bc)

                # Logging
                log_iter = {'iter': i, 'time' : time() - start_time, 'avg_train_loss': train_loss,
                            'validation_loss': validation_loss} # , 'validation_accuracy': validation_accuracy}
                # print(log_iter)
                log.append(log_iter)

                # Early stopping criterion
                if(len(early_stopping_window) == self.__persistence):
                    early_stopping_window = early_stopping_window[1:]
                    early_stopping_window.append(validation_loss)
                    if(min(early_stopping_window) > window_smallest):
                        reached_criterion = True
                        log.append({'early_stop': True})
                        break
                    window_smallest = min(early_stopping_window)
                else:
                    early_stopping_window.append(validation_loss)
        return log

    def step(self, data, w_bc):
        '''
        Calculates the gradient and train loss. Add regularizer to the train loss
        '''
        calculate_grad_loss = self.calculate_grad_loss
        gradient, train_loss = data.map(lambda x: calculate_grad_loss(
            x[0], x[1], w_bc)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        train_loss /= data.count()
        # gradient /= data.count()

        return gradient, train_loss  # + self.l2_reg(w_bc)

    def calculate_grad_loss(self, x, label, w_bc):
        ''' Helper for step, computes the hinge loss and gradient of a single point '''
        xw = x.dot(w_bc.value)[0]
        if self.misclassification(xw, label):
            delta_w = self.gradient(x, label, w_bc=w_bc)
        else:
            delta_w = self.reg_gradient(w_bc, x)
        return delta_w, self.loss_point(x, label, xw=xw, w_bc=w_bc)
        # if self.misclassification(xw, label):
        #     return self.gradient(x, label), self.loss_point(x, label, xw=xw)
        # else:
        #     return 0, 0

    def loss_point(self, x, label, xw=None, w_bc=None):
        ''' Computes the loss of a single point'''
        if xw is None:
            xw = x.dot(w_bc.value)[0]
        return max(1 - label * xw, 0) + self.l2_reg(w_bc, x=x)
        # return max(1 - label * xw, 0)

    def loss(self, data, w_bc=None, spark=None):
        ''' Computes the avg loss (incl regulizer) for a data set '''
        loss_point = self.loss_point
        if w_bc is None and spark is None:
            raise ValueError('w_bc and spark can\'t be None')
        if spark is not None:
            w_bc = spark.sparkContext.broadcast(self.__w)
        return data.map(lambda x: loss_point(x[0], x[1], w_bc=w_bc)).reduce(add)/data.count()
        # return data.map(lambda x: loss_point(x[0], x[1], w_bc=w_bc)).reduce(add) + self.l2_reg(w_bc)

    def l2_reg(self, w_bc, x=None):
        ''' Returns the regularization term '''
        return self.__lambda_reg * (w_bc.value[x.indices] ** 2).sum()/x.nnz
        # return self.__lambda_reg * (w_bc.value ** 2).sum()

    def l2_reg_grad(self, w_bc, x=None):
        '''Returns the gradient of the regularization term  '''
        return 2 * self.__lambda_reg * w_bc.value[x.indices].sum()/x.nnz
        # return 2 * self.__lambda_reg * w_bc.value

    def gradient(self, x, label, w_bc=None):
        ''' Returns the gradient of the loss with respect to the weights '''
        grad = x.copy() * label
        grad.data -= self.l2_reg_grad(w_bc, x)
        return grad
        # return -x*label

    def reg_gradient(self, w_bc, x):
        ''' Sparse matrice loss, gradient of regularizer '''
        return csr_matrix((np.array([-self.l2_reg_grad(w_bc, x)]*x.nnz), x.indices, x.indptr), (1, self.__dim))

    def misclassification(self, x_dot_w, label):
        ''' Returns true if, for a given point, its hingeloss would be > 0. '''
        return x_dot_w * label < 1

    def predict(self, data, w_bc=None, spark=None):
        ''' Predict the labels of the input data '''
        def sign(x): return 1 if x > 0 else -1 if x < 0 else 0
        if w_bc is None and spark is None:
            raise ValueError('w_bc and spark can\'t be None')
        if spark is not None:
            w_bc = spark.sparkContext.broadcast(self.__w)
        return data.map(lambda x: sign(x[0].dot(w_bc.value)) == x[1]).reduce(add)/data.count()
