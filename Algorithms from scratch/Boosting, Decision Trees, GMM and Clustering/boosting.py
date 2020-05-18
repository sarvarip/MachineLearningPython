import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T

		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples

		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################

		total = 0

		for i in range(self.T):
			total += np.multiply(self.clfs_picked[i].predict(features),self.betas[i])

		return np.sign(total)


class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples

		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################

		n = len(features)
		D = (1/n)*np.ones(n)

		for i in range(self.T):
			rate = []
			for clf in self.clfs:
				preds = clf.predict(features)
				rate.append(np.sum(np.multiply(preds!=labels, D)))
			bestidx = np.argmin(rate)
			j = 0
			for clf in self.clfs:
				if j == bestidx:
					chosen_clf = clf
				j = j+1
			self.clfs_picked.append(chosen_clf)
			chosen_rate = rate[bestidx]
			beta = 1/2*np.log((1-chosen_rate)/chosen_rate)
			self.betas.append(beta)
			preds = chosen_clf.predict(features)
			mask = (preds==labels)
			D[mask] = D[mask]*np.exp(-beta)
			D[~mask] = D[~mask]*np.exp(beta)
			D = D/np.sum(D)

	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
