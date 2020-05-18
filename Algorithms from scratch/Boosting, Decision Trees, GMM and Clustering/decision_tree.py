import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return

	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')

		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		self.features = np.array(self.features)

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2 or self.features.shape[1] == 0:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array,
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of
					  corresponding training samples
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●

				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################

			braches = np.array(branches)
			B = branches.shape[1]
			portions = np.sum(branches, axis=0)
			#print("Branches")
			#print(braches)
			portions = portions / np.sum(branches)

			entropy = []
			for i in range(B):
				dist = braches[:,i] / np.sum(braches[:,i])
				dist = np.nan_to_num(dist) #if denominator is zero, just leave it as zero
				logdist = np.log2(dist)
				logdist = np.nan_to_num(logdist)
				ent = -np.sum(np.multiply(dist, logdist))
				entropy.append(ent)
				#print(ent)
				#print(entropy)
			cond_entropy = np.sum(np.multiply(portions, entropy))
			#print(portions)
			#print(entropy)
			#print("Cond entropy")
			#print(cond_entropy)

			return cond_entropy



		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################

		cond_entropies = []
		for idx_dim in range(self.features.shape[1]):
			self.feature_uniq_split = np.unique(self.features[:,idx_dim])
			C = self.num_cls
			#print("C and unique labels")
			#print(C)
			#print(np.unique(self.labels))
			B = len(self.feature_uniq_split)
			braches = np.zeros((C,B))
			i = 0
			for label in np.unique(self.labels):
				j = 0
				for value in self.feature_uniq_split:
					braches[i, j] = np.sum(np.multiply(self.features[:,idx_dim]==value, np.equal(self.labels, label)))
					j = j+1
				i = i+1
			cond_entropies = conditional_entropy(braches)
		best_idx = np.argmin(cond_entropies)
		self.dim_split = best_idx


		############################################################
		# TODO: split the node, add child nodes
		############################################################

		values = np.unique(self.features[:,best_idx])
		for value in values:
			mask = (self.features[:,best_idx] == value)
			newfeat = np.delete(self.features, best_idx, 1)
			newfeat = newfeat[mask,]
			newlabs = [label for idx, label in enumerate(self.labels) if mask[idx]]
			numclass = len(np.unique(newlabs))
			self.children.append(TreeNode(newfeat, newlabs, numclass))

		# split the child nodes
		#print("Children length")
		#print(len(self.children))
		for child in self.children:
			#print("Child labels")
			#print(child.labels)
			#print("Child features")
			#print(child.features)
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			if isinstance(self.feature_uniq_split, np.ndarray):
				self.feature_uniq_split = self.feature_uniq_split.tolist()
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])

			#Needed to add this myself, otherwise wrong accuracy prediction,
			#because indexing becomes incorrect
			feature = np.delete(feature, self.dim_split)

			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max
