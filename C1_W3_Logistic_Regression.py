import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math


# load dataset
X_train, y_train = load_data("ex2data1.txt")



def compute_cost(X, y, w, b, lambda_= 1):
 m, n = X.shape

 ### START CODE HERE ###
 loss_sum = 0 

 # Loop over each training example
 for i in range(m): 

     # First calculate z_wb = w[0]*X[i][0]+...+w[n-1]*X[i][n-1]+b
     z_wb = 0 
     # Loop over each feature
     for j in range(n): 
         # Add the corresponding term to z_wb
         z_wb_ij = w[j]*X[i][j] # Your code here to calculate w[j] * X[i][j]
         z_wb += z_wb_ij # equivalent to z_wb = z_wb + z_wb_ij
     # Add the bias term to z_wb
     z_wb += b # equivalent to z_wb = z_wb + b

     f_wb = sig(z_wb) # Your code here to calculate prediction f_wb for a training example
     loss =  -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)

     loss_sum += loss # equivalent to loss_sum = loss_sum + loss

 total_cost = (1 / m) * loss_sum  
 ### END CODE HERE ### 

 return total_cost


def compute_gradient(X, y, w, b, lambda_=None): 
          m, n = X.shape
          dj_dw = np.zeros(w.shape)
          dj_db = 0.

          ### START CODE HERE ### 
          for i in range(m):
             z_wb = 0
             # Loop over each feature
             for j in range(n): 
                     # Add the corresponding term to z_wb
                     z_wb_ij = X[i, j] * w[j]
                     z_wb += z_wb_ij

             # Add bias term 
             z_wb += b

             # Calculate the prediction from the model
             f_wb = sig(z_wb)

              # Calculate the  gradient for b from this example
             dj_db_i = f_wb - y[i] # Your code here to calculate the error

              # add that to dj_db
             dj_db += dj_db_i

              # get dj_dw for each attribute
             for j in range(n):
                  # You code here to calculate the gradient from the i-th example for j-th attribute
                  dj_dw_ij = (f_wb - y[i])* X[i][j]  
                  dj_dw[j] += dj_dw_ij

          # divide dj_db and dj_dw by total number of examples
          dj_dw = dj_dw / m
          dj_db = dj_db / m
          ### END CODE HERE ###

          return dj_db, dj_dw
      
 def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant
      
    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing


def predict(X, w, b): 
          # number of training examples
          m, n = X.shape   
          p = np.zeros(m)

          ### START CODE HERE ### 
          # Loop over each example
          for i in range(m):   
            # Calculate f_wb (exactly how you did it in the compute_cost function above)
            z_wb = 0
            # Loop over each feature
            for j in range(n): 
                     # Add the corresponding term to z_wb
                     z_wb_ij = X[i, j] * w[j]
                     z_wb += z_wb_ij

            # Add bias term 
            z_wb += b

            # Calculate the prediction from the model
            f_wb = sig(z_wb)
            p[i] = f_wb >= 0.5
          ### END CODE HERE ### 
          return p

def compute_cost_reg(X, y, w, b, lambda_ = 1):

         m, n = X.shape

          # Calls the compute_cost function that you implemented above
         cost_without_reg = compute_cost(X, y, w, b) 

          # You need to calculate this value
         reg_cost = 0.

          ### START CODE HERE ###
         for j in range(n):
              reg_cost_j = w[j]**2
              reg_cost = reg_cost + reg_cost_j

          ### END CODE HERE ### 

          # Add the regularization cost to get the total cost
         total_cost = cost_without_reg + (lambda_/(2 * m)) * reg_cost

         return total_cost
    
    
def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
      m, n = X.shape

      dj_db, dj_dw = compute_gradient(X, y, w, b)

      ### START CODE HERE ###     
      # Loop over the elements of w
      for j in range(n): 

          dj_dw_j_reg = (lambda_ / m) * w[j] 

          # Add the regularization term  to the correspoding element of dj_dw
          dj_dw[j] = dj_dw[j] + dj_dw_j_reg

      ### END CODE HERE ###         

      return dj_db, dj_dw
