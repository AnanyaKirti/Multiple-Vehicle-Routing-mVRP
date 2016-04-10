#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Vechicle Routing Problem
=========================================================



"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
from sklearn import datasets


def gen_data(number, dimension):
	"""
	function to generate the location of the nodes.
	"""
	return dimension * np.random.random_sample((number, 2)) + dimension



def gen_cost(number, maximum):
	"""
	function to generate the cost of traveling from a node to another.
	"""
	dummy =  maximum * np.random.random_sample((number, number)) + maximum

	return np.add(dummy , dummy.transpose() )
	

if __name__ == '__main__':
	

	iris = datasets.load_iris()

	# X = iris.data
	# y = iris.target
	# final = KMeans(n_clusters=3).fit_predict(X)
	# print y
	# print final
	data =  gen_data(10, 15)
	cost_matrix = gen_cost(10, 40)

	final = KMeans(n_clusters=3).fit_predict(data)
	print final