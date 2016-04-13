#!/usr/bin/python
# -*- coding: utf-8 -*-



from __future__ import print_function
import math
import random
from simanneal import Annealer

"""
=========================================================
Vechicle Routing Problem
=========================================================



"""
# print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.cluster import KMeans
from sklearn import datasets

import time





class TravellingSalesmanProblem(Annealer):

	"""
	Test annealer with a travelling salesman problem.
	"""
	
	# pass extra data (the distance matrix) into the constructor
	def __init__(self, state, distance_matrix):
		self.distance_matrix = distance_matrix
		super(TravellingSalesmanProblem, self).__init__(state)  # important! 

	def move(self):
		"""Swaps two cities in the route."""
		a = random.randint(0, len(self.state) - 1)
		b = random.randint(0, len(self.state) - 1)
		self.state[a], self.state[b] = self.state[b], self.state[a]

	def energy(self):
		"""Calculates the length of the route."""
		e = 0
		for i in range(0,len(self.state)):
			e += self.distance_matrix[i-1][i]
		return e

def annealer_gen(data_cluster , data_cluster_index):
	"""
	Driver function for annealer per cluster.
	"""
	init_state = data_cluster_index
	random.shuffle(init_state)

	# create a distance matrix

	distance_matrix = gen_cost(data_cluster)
	tsp = TravellingSalesmanProblem(init_state, distance_matrix)
	state, e = tsp.anneal()
	print(state)
	print(e)


def gen_data(number, dimension):
	"""
	function to generate the location of the nodes.
	@param number : number of data to be produced
	@param dimension : limit the data in the given dimension. eg 35x35 units.
	Output: returns the random sample dataset.
	"""
	return dimension * np.random.random_sample((number, 2)) 



def gen_cost(data):
	"""
	function to generate the cost of traveling from a node to another.
	The cost is defined as the distance between the two nodes.
	@param data : the randomly generated data_set.
	Output the required cost of traversing the nodes.
	"""
	return squareform(pdist(data, 'euclidean'))

def slice_data(data, cluster, no_of_clusters):
	"""
	function to slice the matrix according to the cluster.
	Cluster -> data_index -> dimension
	@param data : the randomly generated data_set.
	@param cluster : the KMeans cluster which has been computed.
	@param no_of_clusters : the number of clusters used in KMeans
	Output : slice_data
	"""
	dummy = [[] for i in range(0,no_of_clusters)]
	dummy_index = [[] for i in range(0,no_of_clusters)]
	j = 0
	for i in cluster:
		dummy[i].append(data[j])
		dummy_index[i].append(j)
		j +=1
	return dummy, dummy_index
		





def main():
	# iris = datasets.load_iris()

	# X = iris.data
	# y = iris.target
	# final = KMeans(n_clusters=3).fit_predict(X)
	# print y
	# print final

	# generate the data
	no_data = 10
	dimension = 15
	data =  gen_data(no_data, dimension)
	# generate the cost matrix data
	distance_matrix = gen_cost(data)
	no_of_clusters = 2		# number of clusters.
	# call KMeans clustering
	cluster = KMeans(n_clusters=no_of_clusters).fit_predict(data)
	# Slice the data.
	data_cluster, data_cluster_index = slice_data(data, cluster, no_of_clusters)
	print (data)
	# print (distance_matrix)
	# print (cluster)
	print (data_cluster_index)
	for i in range(0,no_of_clusters):
		# print(i)
		annealer_gen(data_cluster[i], data_cluster_index[i])

	
	


if __name__ == '__main__':
	start = time.clock()
	main()
	print (time.clock() - start , "seconds")