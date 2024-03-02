# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:24:13 2023

@author: xyu1
"""

import matplotlib.pyplot as plt
from autograd import grad, value_and_grad
import autograd.numpy as np
from functools import partial
from autograd.scipy.stats import t, norm, multivariate_normal
from scipy.special import loggamma
from autograd.core import getval
from autograd.numpy import random as npr
from autograd.misc.optimizers import adam
from scipy.stats import multivariate_normal as mvn
from autograd.scipy.stats import multivariate_normal
import autograd.scipy.stats.t as t_dist

import matplotlib.animation as animation
import matplotlib.image as mpimg
import os

from autograd.scipy.special import logsumexp


#%%
class Model():

	def __init__(self, mean1, mean2, cov):

		self.mean1 = mean1
		self.mean2 = mean2
		self.cov = cov
		assert self.mean1.shape[0] == self.cov.shape[0]
		assert self.mean2.shape[0] == self.cov.shape[0]
		self.d = self.cov.shape[0]


	def log_density(self, zs):
		
		clust1_density = 0.5 * multivariate_normal.logpdf(zs, self.mean1, cov=self.cov)
		clust2_density = 0.5 * multivariate_normal.logpdf(zs, self.mean2, cov=self.cov)
		return np.logaddexp(clust1_density, clust2_density)

	def sample(self, n):
		num_clust1 = np.random.binomial(n=n, p=0.5)
		num_clust2 = n - num_clust1
		samples_mode1 = mvn.rvs(mean=self.mean1, cov=self.cov, size=num_clust1)
		samples_mode2 = mvn.rvs(mean=self.mean2, cov=self.cov, size=num_clust2)
		samples = np.vstack([samples_mode1, samples_mode2])
		return samples


class ApproxMFGaussian():

	def __init__(self):
		pass

	def log_density(self, var_param, z):
		# variational density evaluated at samples
		return multivariate_normal.logpdf(z, var_param[:2], np.diag(np.exp(var_param[2:] * 2)))

	def sample(self, var_param, S):
		stddevs = np.exp(var_param[2:])
		return var_param[:2] + seed.randn(S, 2) * np.expand_dims(stddevs, 0)

	def gradq(self, var_param, z):
		objective = lambda vparam : self.log_density(vparam, z)
		grad_q = grad(objective)
		return grad_q(var_param)

	def entropy(self, var_param):
		return 0.5 * 2 * (1.0 + np.log(2*np.pi)) + np.sum(var_param[2:] * 2)
		
#%%
d = 2

# Generate data
mean1 = np.array([-4, -4])
mean2 = np.array([4, 4])
cov = np.eye(d)

model = Model(mean1=mean1, mean2=mean2, cov=cov)
approx = ApproxMFGaussian()

S = 1000

# Variational parameters: first d elements are the mean, last d elements are the diagonal of log-covariance
variational_mean = np.array([0, 0])
variational_log_cov_diag = np.ones(d)
variational_param = np.concatenate([variational_mean, variational_log_cov_diag])

#%%
def objective(variational_param, iter):

    samples = approx.sample(variational_param, S)

    lik = np.mean(model.log_density(samples))
    entropy = approx.entropy(variational_param)
    elbo = lik + entropy

    return -elbo / S

seed = npr.RandomState(0)


def plot_isocontours(ax, func, xlimits=[-10, 10], ylimits=[-10, 10], numticks=101, cmap='viridis', label=''):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)

    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, cmap=cmap)
    plt.savefig('before.pdf')
    ax.set_yticks([])
    ax.set_xticks([])

    
    

fig = plt.figure(figsize=(10, 10), dpi=200,facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=True)
target_distribution = lambda x : np.exp(model.log_density(x))
plot_isocontours(ax, target_distribution, label="p(z)")
#plot_isocontours(ax, variational_contour, cmap='plasma', label="q(z)")
plt.title('Before optimization')
plt.savefig('kl_before.pdf')
#%%
plt.close()

fig = plt.figure(figsize=(10, 10), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=True)

#%%
imgs = []
print("     Epoch     |    Objective     |    Param vals")
def print_perf(params, iter, grad):
	
    bound = objective(params, iter)
    message = "{:15}|{:20}|{:15}|{:15}{:15}|{:15}".format(iter, round(bound, 2), round(float(params[0]), 2), round(float(params[1]), 2), round(float(params[2]), 2), round(float(params[3]), 2))
    print(message)

    plt.cla()
    target_distribution = lambda x : np.exp(model.log_density(x))
    plot_isocontours(ax, target_distribution, label="p(z)")

    variational_contour = lambda x: mvn.pdf(x, params[:2], np.diag(np.exp(params[2:])))
    plot_isocontours(ax, variational_contour, cmap='plasma', label="q(z)")
    plt.title('After optimization with KL divergence')
    plt.savefig('after_figure.pdf',bbox_inches='tight')
   


num_epochs = 100
step_size = .1

optimized_params = adam(grad=grad(objective), x0=variational_param, 
                        step_size=step_size, num_iters=num_epochs, 
                        callback=print_perf)









    