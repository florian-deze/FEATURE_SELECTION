"""
	define functions with all used models in feature selection project

	model used:
		RandomForestclassifier
		LightGbm
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#import lgbm

########################################
# prepare datasets with adapted format #
########################################

def prepareLgbm(train_x, train_y):
	#return train_x, train_y, val_x, val_y
	pass


################
# train models #
################

def trainLgbm(train_x, train_y):
	train_x, train_y, val_x, val_y = prepareLgbm(train_x, train_y)
	pass

def trainRandomForestClassifier(train_x, train_y):
	model = RandomForestClassifier()
	model.fit(train_x, train_y)
	return model


#################################
# restitution in correct format #
#################################

def formatRandomForestClassifier(model, columns):
	result = pd.DataFrame(columns=["FEAT", "SCORE"])
	result["FEAT"] = columns
	result["SCORE"] = model.feature_importances_
	return result


###################################
# adversarial validation tps func #
###################################

def tpsRandomForestClassifier(train_x: pd.DataFrame, train_y:pd.Series):
	model = trainRandomForestClassifier(train_x, train_y)
	return formatRandomForestClassifier(model, train_x.columns)


