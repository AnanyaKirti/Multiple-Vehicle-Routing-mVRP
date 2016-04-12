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
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.cluster import KMeans
from sklearn import datasets


def gen_data(number, dimension):
	"""
	function to generate the location of the nodes.
	"""
	return dimension * np.random.random_sample((number, 2)) 



def gen_cost(data):
	"""
	function to generate the cost of traveling from a node to another.
	The cost is defined as the distance between the two nodes.
	"""

	return squareform(pdist(data, 'euclidean'))
	


def main():
	iris = datasets.load_iris()

	# X = iris.data
	# y = iris.target
	# final = KMeans(n_clusters=3).fit_predict(X)
	# print y
	# print final
	data =  gen_data(10, 15)
	print data
	cost_matrix = gen_cost(data)

	final = KMeans(n_clusters=2).fit_predict(data)
	print final



if __name__ == '__main__':
	main()
	