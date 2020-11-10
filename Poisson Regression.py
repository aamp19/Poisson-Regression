import numpy as np
import util
import pandas as pd
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    # *** END CODE HERE ***
    valid_data = pd.read_csv(eval_path)
    valid_x = np.array(valid_data[["x_1","x_2","x_3","x_4"]].values)
    valid_y = np.array(valid_data[["y"]].values)
   
    train = pd.read_csv(train_path)
    train_x = np.array(train[["x_1","x_2","x_3","x_4"]].values)
    train_y = np.array(train[["y"]].values)
    
    poisson_model = PoissonRegression(step_size=lr)
    poisson_model.fit(train_x, train_y)
    
    
    y = poisson_model.predict(valid_x)
    plt.figure()
    print('valid_y', valid_y.shape)
    print('y ',y.shape)
    plt.scatter(valid_y, y, alpha=0.4, c='red', label='Ground Truth vs Predicted')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.legend()
    plt.savefig('poisson_valid.png')


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=np.array([0.00001,0.0001,0.00001,0.0001]), verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        #self.theta = self.theta.reshape(len(self.theta),1)
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        #print('theta shape',self.theta.shape)
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n, dtype=np.float32)

        
        for i in range(m):
#             self.theta += self.theta + self.step_size * (np.exp(x[i].dot(np.transpose(self.theta))) - y[i]).dot(x[i])
#             self.theta +=  self.step_size*(np.dot(np.exp(np.dot(self.theta.transpose(), x[i].reshape(len(x[i]),1))) - y[i], x[i].reshape(1,len(x[i])))).transpose()
#             self.theta += self.theta + self.step_size * (np.exp(x.dot(np.transpose(self.theta)) - y)).dot(x)
            #theta = self.theta.reshape(len(self.theta),1)
            exponent = self.step_size*(np.exp(np.dot(self.theta.reshape(1,len(self.theta)),x[i].reshape(len(x[i]),1)))) - y[i]
#             print("exponent ",exponent.shape)
#             print('x ',x[i].reshape(len(x[i]),1).shape)
            dot = exponent*x[i].reshape(len(x[i]),1)
#             print('dot ',dot.shape)
#             print('x',x[i].shape)
#             print('y',y[i].shape)
#             print('theta',self.theta.shape)
            #self.theta = self.theta.reshape(len(self.theta),1) + dot
            self.theta += (self.step_size * (np.exp(np.dot(self.theta,x[i])) - y[i]) * x[i])*-1
            #print('theta size',self.theta.shape)
#             self.theta += 10**-5(np.exp((np.dot(self.theta.reshape(1,len(self.theta)),x[i].reshape(len(x[i]),1)))) - y[i])*x[i].reshape(len(x[i]),1)
            
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        #print("x", x.shape)
        #print(self.theta.shape)
        #print('theta ',self.theta.shape)
       # print('x shape', x.shape)
        #y_hat = np.exp(np.dot(self.theta.transpose(), x.reshape(4,len(x))))
        #print('self.theta', self.theta.shape)
        #print('x ',x.shape)
        y_hat = np.exp(np.dot(self.theta.transpose(),x.transpose()))
        
        #yhat = np.exp(np.dot(self.theta.reshape(1,len(self.theta)), x.transpose()))
        return y_hat
        #print('y_hat',y_hat.shape)
#         yhat = np.exp(np.dot(self.theta.transpose().reshape(len(self.theta),1),x.transpose()))
        #return yhat
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.png')
