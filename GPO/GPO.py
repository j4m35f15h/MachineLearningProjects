# -*- coding: utf-8 -*-
"""
Gaussian Process Optimizer Module

This module implements a Gaussian Process (GP) optimizer designed to maximize
an unknown objective function through sequential sampling. The optimizer uses 
various acquisition functions to balance exploration and exploitation in the 
search for an optimal solution.

Key Components:
- Kernel Function:
    - `RBF_kernel`: Implements the Radial Basis Function (RBF) kernel, used to
        model the similarity between data points based on their distance in the
        input space. The RBF kernel is parameterized by a signal variance 
        (`sigma`) and a length scale (`len_scale`).

- Acquisition Functions:
    - `PI_aquisition`: Probability of Improvement (PI), which prioritizes 
        points that are likely to exceed the current maximum observed value.
    - `EI_aquisition`: Expected Improvement (EI), which balances exploration 
        and exploitation by considering both the improvement magnitude and 
        probability.
    - `UCB_aquisition`: Upper Confidence Bound (UCB), which selects points 
        based on an upper bound of the mean and variance, weighted by a scaling 
        parameter (`lambda_scale`).
    - `maxVar_aquisition`: Maximizes the variance, selecting points with the 
        highest uncertainty for exploration.
    - `upperEst_aquisition`: Chooses points with the highest mean estimate, 
        focusing on exploitation.

- GP Optimizer Function:
    - `GPOptimiser`: The main optimization function. It takes a user-defined 
        objective function and domain parameters, initializes a set of sampled 
        points, and iteratively selects new sampling points based on the 
        specified acquisition function. Optional plotting allows for 
        visualization of the GP’s progress over the domain.

Example Usage:
    The module can be used to find the maximum of a custom function over a 
    specified domain by defining the objective function, setting initial sample
    points, and calling `GPOptimiser` with desired parameters.

Author: James
Created: October, 2024
"""

import math
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

#Kernel functions for preduction ---------------------------------
def RBF_kernel(x1,x2,sigma = 2,len_scale = 1):
    """
    Radial Basis Function (RBF) kernel, also known as Gaussian kernel.
    Computes a measure of similarity between two input points x1 and x2.
    
    Parameters:
    x1, x2 : Input points.
    sigma : Variance parameter.
    len_scale : Length scale parameter.
    
    Returns:
    The RBF kernel matrix.
    """
    
    sq_dist = x1**2+(x2**2).T - 2*x1*x2.T
    return sigma**2 * np.exp(-sq_dist/(2*len_scale**2))


#Aquisition functions to explore the function domain -------------


def PI_aquisition(x_axis,mu_est,var_est,y_vals_max,x_vals_index = None):
    """
    Probability of Improvement (PI) acquisition function.
    Calculates the probability of an improvement over the current maximum.
    
    Parameters:
    x_axis : Range of x values to evaluate.
    mu_est, var_est : Mean and variance estimates.
    y_vals_max : Current maximum y value observed.
    x_vals_index : Index of values already evaluated.
    
    Returns:
    Index of the next point to sample and its corresponding x value.
    """
    
    aquisition_axis = [[1-norm.cdf(y_vals_max,mu_est[i], var_est.T[i])] for i in range(len(x_axis))][0][0]    
    print(x_vals_index.shape)
    if x_vals_index.shape[1] == 0:
        for i in x_vals_index[0]:
            aquisition_axis[i] = 0
        
    x_new_index = np.argmax(aquisition_axis)
    return x_new_index,x_axis[0,x_new_index]


def EI_aquisition(x_axis,mu_est,var_est,y_vals_max,x_vals_index = None):
    """
    Expected Improvement (EI) acquisition function.
    Calculates the expected improvement over the current maximum.
    
    Parameters:
    x_axis : Range of x values to evaluate.
    mu_est, var_est : Mean and variance estimates.
    y_vals_max : Current maximum y value observed.
    x_vals_index : Index of values already evaluated.
    
    Returns:
    Index of the next point to sample and its corresponding x value.
    """
    aquisition_axis = [[ -1*(y_vals_max-mu_est[i])*(1-norm.cdf(mu_est[i],y_vals_max, var_est.T[i])) + var_est.T[i]*(norm.pdf(mu_est[i],y_vals_max, var_est.T[i]))       ] for i in range(len(x_axis))][0][0]
    
    print("p2",x_vals_index.shape)
    if x_vals_index.shape[1] == 0:
        for i in x_vals_index[0]:
            aquisition_axis[i] = float('-Inf')
    
    x_new_index = np.argmax(aquisition_axis)
    return x_new_index,x_axis[0,x_new_index]

def UCB_aquisition(x_axis,mu_est,var_est,lambda_scale = 1):
    """
    Upper Confidence Bound (UCB) acquisition function.
    Balances exploration and exploitation by adjusting the weight of variance.
    
    Parameters:
    x_axis : Range of x values to evaluate.
    mu_est, var_est : Mean and variance estimates.
    lambda_scale : Scale parameter for exploration.
    
    Returns:
    Index of the next point to sample and its corresponding x value.
    """
    
    aquisition_axis = mu_est + lambda_scale*var_est.T

    x_new_index = np.argmax(aquisition_axis)
    return x_new_index,x_axis[0,x_new_index]


def maxVar_aquisition(x_axis,var_est):
    """
    Maximizes the variance to explore areas of high uncertainty.
    
    Returns:
    Index of the next point to sample and its corresponding x value.
    """
    
    x_new_index = np.argmax(var_est.T)
    return x_new_index,x_axis[0,x_new_index]

def upperEst_aquisition(x_axis,upper_est):
    """
    Maximizes the upper bound estimate, focusing on exploitation.
    
    Returns:
    Index of the next point to sample and its corresponding x value.
    """
    
    x_new_index = np.argmax(upper_est)
    return x_new_index,x_axis[0,x_new_index]


#Main Optimiser --------------------------------------------------


def GPOptimiser(x_domain_start,x_domain_end,x_domain_stepCount,functionInput,initialXVals,initialYVals,aquisition_function = 'UCB',sigma = 2,len_scale = 1,plotOpt = False,maxIter = 10,sigmaN = 0,lambda_scale = 1):
    """
    Gaussian Process Optimizer with configurable acquisition functions.

    Parameters:
    - x_domain_start, x_domain_end: Define the range of the input domain.
    - x_domain_stepCount: Number of points in the input domain.
    - functionInput: Objective function to be optimized.
    - initialXVals, initialYVals: Initial samples for x and their corresponding function values.
    - aquisition_function: Acquisition function to guide exploration/exploitation ('PI', 'EI', 'UCB', 'max_var', 'max_est').
    - sigma: Signal variance for the RBF kernel.
    - len_scale: Length scale for the RBF kernel.
    - plotOpt: Boolean to enable plotting of predictions during optimization.
    - maxIter: Maximum iterations for the optimization loop.
    - sigmaN: Noise variance for the GP model.
    - lambda_scale: Exploration-exploitation trade-off parameter for UCB.

    Returns:
    - The maximum value of the objective function found by the optimizer.
    """
    # Generate evenly spaced points in the input domain
    x_axis = np.linspace(x_domain_start, x_domain_end, x_domain_stepCount).reshape(1, -1)
    y_vals = initialYVals  # Function values at sampled points
    x_vals = initialXVals  # Sampled points in the input domain
    x_vals_index = np.array([[]])  # Indices of sampled points
    
    
    for _ in range(maxIter):
        
        # Identify the current maximum function value
        max_index = y_vals.T.argmax()
        y_vals_max = y_vals.T[max_index]
        
        # Compute Gaussian Process kernel matrices for model inference
        axAx_kern   = RBF_kernel(x_axis,x_axis,sigma,len_scale)
        axVal_kern  = RBF_kernel(x_axis,x_vals,sigma,len_scale)
        valVal_kern = RBF_kernel(x_vals,x_vals,sigma,len_scale)
        
        # Estimate the mean function values at all points in the input domain
        mu_est = axVal_kern.T.dot(np.linalg.inv(valVal_kern)).dot(y_vals.T)
        mu_est = mu_est.T
        
        # Estimate the variance at each point in the input domain
        var_est = np.diag(axAx_kern - axVal_kern.T.dot(np.linalg.inv(valVal_kern)).dot(axVal_kern)).reshape(-1,1) + sigmaN
        
        # Calculate the upper and lower bounds of confidence intervals
        upper_est = mu_est + var_est.T
        lower_est = mu_est - var_est.T
        
        # Optional: Plot the current mean estimate, bounds, and sampled points
        if plotOpt:
            fig,ax = plt.subplots()
            current_est_line = ax.plot(x_axis[0],mu_est[0])
            current_est_points = ax.scatter(x_vals[0],y_vals[0])
            upper_est_line = ax.plot(x_axis[0],upper_est[0])
            lower_est_line = ax.plot(x_axis[0],lower_est[0])
        
        
        # Select the acquisition function to compute the next point to sample
        if aquisition_function == 'PI':
            x_new_index,x_new = PI_aquisition(x_axis,mu_est,var_est,y_vals_max,x_vals_index)
        elif aquisition_function == 'EI':
            x_new_index,x_new = EI_aquisition(x_axis,mu_est,var_est,y_vals_max,x_vals_index)
        elif aquisition_function == 'UCB':
            x_new_index,x_new = UCB_aquisition(x_axis,mu_est,var_est,lambda_scale)
        elif aquisition_function == 'max_var':
            x_new_index,x_new = maxVar_aquisition(x_axis,var_est)
        elif aquisition_function == 'max_est':
            x_new_index,x_new = upperEst_aquisition(x_axis,upper_est)
        
        # Check if the new sample is already in sampled points to avoid resampling
        if x_new in x_vals[0]:
            print("Maximum found at: {}".format(max(y_vals[0])))
            return
        
        # Add the new sample and its index to sampled points
        x_vals = np.concatenate((x_vals[0],[x_new])).reshape(1,-1)
        x_vals_index = np.concatenate((x_vals_index[0],[x_new_index])).reshape(1,-1)
        
        # Evaluate the function at the new sample and add it to function values
        y_vals = np.concatenate((y_vals[0],[functionInput(x_new)])).reshape(1,-1)
    
    print("Maximum found after maximum iterations: {}".format(max(y_vals[0])))
    return max(y_vals[0])


#test Optimisation ----------------------------------------------
   
def main():
    """
    Tests the GP Optimizer with a sample objective function.
    """
    def myObjective(x):
        return x*np.sin(x)

    # Define optimization domain and parameters
    x_domain_start = 0
    x_domain_end = 10
    x_domain_stepCount = 100
    functionInput = myObjective
    initialXVals = np.array([[2.45,3.43,4.54,6.5,8.8]])
    initialYVals = myObjective(initialXVals)
    sigma = 2
    len_scale = 1
    plotOpt = True
    maxIter = 10
    sigmaN = 0.5
    lambda_scale = 10
    
    fig,ax = plt.subplots() 
    
    
    x_gt  = np.linspace(x_domain_start,x_domain_end,x_domain_stepCount).reshape(1,-1)
    y_gt = myObjective(x_gt)
    ground_truth = ax.plot(x_gt[0],y_gt[0],label = 'Ground truth')
    
    # Run optimizer and display result
    maximised_function = GPOptimiser(x_domain_start,x_domain_end,x_domain_stepCount,functionInput,initialXVals,initialYVals,sigma,len_scale,plotOpt,maxIter,sigmaN,lambda_scale)
    print("Optimiser returned: {}".format(maximised_function))
   
    
if __name__ == "__main__":
    main()