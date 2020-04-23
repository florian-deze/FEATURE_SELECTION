import random
import numpy as np
import pandas as pd
import pprint
from sklearn.model_selection import train_test_split

import adversarial_validation as adv
import model

TEST_FRAC = 0.2
np.random.seed(42) # set seed to have the same results

def generateNormalWithRandomParams(max_loc, max_scale, size):
	m = random.uniform(0, max_loc)
	v = random.uniform(0, max_scale)
	return np.random.normal(loc=m, scale=v, size=size)

def generateRandomNormalHomogeneous(homogeneous_size, size_dataset):
	#train = pd.DataFrame(column=["homogeneous_"+str(i) for i in range(0, homogeneous_size)])
	data = pd.concat([pd.Series(generateNormalWithRandomParams(10, 4, size_dataset)) for i in range(0, homogeneous_size)], axis=1)
	data.columns = ["homogeneous_"+str(i) for i in range(0, homogeneous_size)]
	train, test = train_test_split(data, test_size=TEST_FRAC)
	return train, test

def generateRandomNormalNotHomogeneous(degree_of_diff, size_dataset):
	train = pd.DataFrame(columns=["not_homogeneous_"+str(i) for i in range(0, len(degree_of_diff))])
	test = pd.DataFrame(columns=["not_homogeneous_"+str(i) for i in range(0, len(degree_of_diff))])
	test_size = int(size_dataset*TEST_FRAC)
	train_size = int(size_dataset*(1-TEST_FRAC))
	for col, diff in zip(train.columns, degree_of_diff):
		m = random.uniform(0, 10)
		v = random.uniform(0, 4)
		train[col] = np.random.normal(loc=m, scale=v, size=train_size)
		test[col] = np.random.normal(loc=m+diff[0], scale=v+diff[1], size=test_size)
	return train, test

def generateRandomNormal(size_dataset, homogeneous_size, degree_of_diff):
	train_h, test_h= generateRandomNormalHomogeneous(homogeneous_size, size_dataset)
	train, test = generateRandomNormalNotHomogeneous(degree_of_diff, size_dataset)
	train = pd.concat([train_h.reset_index(drop=True), train.reset_index(drop=True)], axis=1)
	test = pd.concat([test_h.reset_index(drop=True), test.reset_index(drop=True)], axis=1)
	return train, test

def demo():
	train, test = generateRandomNormal(1000, 10, [(0.5, 0.1), (1, 1), (-3, 0), (3, 2), (10, 5)])
	result_ks = adv.adversarialValidationKs(train, test, 0.1)

	result, suggestion = adv.adversarialValidation(train, test, [], 0.1, model.tpsRandomForestClassifier)
	print(result_ks.sort_values(by="P_VALUE", ascending=False))
	print(result.sort_values(by="P_VALUE", ascending=False))
	print(suggestion)

demo()