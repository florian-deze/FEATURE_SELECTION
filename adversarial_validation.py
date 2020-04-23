import gc
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


######################################
# ADVERSARIAL VALIDATION USING MODEL #
######################################

TARGET_ADVERSARIAL = "advers_label"


def labelisationAdversarialValidation(train, test):
	train.loc[train.index,TARGET_ADVERSARIAL] = 0
	test.loc[test.index,TARGET_ADVERSARIAL] = 1
	prop = round(test.shape[0] / (train.shape[0] + test.shape[0]),1)
	# suffle entirely
	train = train.append(test)
	train = shuffle(train)
	# suffle in taking a fraction
	train, test = train_test_split(train, test_size=prop)
	return train, test
	

def adversarialValidationModel(data:pd.DataFrame, target:str, tps_func):
	# prepare
	train_x = data.drop(labels=target, axis=1)
	train_y = data[target]
	# save memory space
	del data
	gc.collect()
	# run train/pred/score func
	total_vars = train_x.shape[1]
	nb_vars = total_vars
	result = pd.DataFrame(columns=["FEAT", "SCORE"])
	while nb_vars>1:
		score = tps_func(train_x, train_y)
		score = score.sort_values(by="SCORE", ascending=False)
		result = result.append(score.iloc[0].T)
		train_x = train_x.drop(labels=score.iloc[0]["FEAT"], axis=1)
		nb_vars=train_x.shape[1]
		if nb_vars<2:
			result = result.append(score.iloc[1].T)
	step = 1/total_vars
	result["SCORE"] = [round(1-i*step,2) for i in range(0, total_vars)]
	# the best score mean the variable is less significative to determine train from test
	result["SCORE"] = 1-result["SCORE"]
	return result


###################################
# ADVERSARIAL VALIDATION USING KS #
###################################


def ks_test_per_feat(train, test):
	return {c:[round(ks_2samp(train[c], test[c])[1],2)] for c in test.columns}

def adversarialValidationKs(train:pd.DataFrame, test:pd.DataFrame):
	result = pd.DataFrame().from_dict(ks_test_per_feat(train, test)).reset_index(drop=True)
	result = result.T.reset_index()
	result.columns = ["FEAT", "P_VALUE"]
	return result


#################################
# ADVERSARIAL VALIDATION GLOBAL #
#################################


def adversarialValidation(train:pd.DataFrame, test:pd.DataFrame, categorical:list, threshold_ks, tps_func):
	"""
	adversarial validation optimized in time
	DESCRIPTION:
		Use kolmogorov smirnov with monte carlos simulation on all feats 
		and get mean(p_value) for each of them
		For all p_value > threshold_ks*1.5 we considere the 
		probability high enough to not be tested with a model
		Then we make a test base on a model for the remaining values
	INPUT:
		train/test :: pd.DataFrame :: train and test dataset of your project
		categorical :: list :: list of al categorical features
		threshold_ks :: float :: minimal value to accept H0 in ks test
		tps_func :: function :: train, test and score the most useful 
								vars in deteting the train from the test
	OUTPUT:
		rank :: pd.DataFrame :: the ranking of the feats from best to worst
		suggestion :: list :: the list of feats to keep
	"""

	# adversarial validation with ks test
	result_ks = adversarialValidationKs(train.drop(labels=categorical, axis=1), test.drop(labels=categorical, axis=1))
	# remove feats with high p_value or p_value at 0
	feats = result_ks[result_ks["P_VALUE"]<threshold_ks*2.5]
	feats = feats[feats["P_VALUE"]>0]["FEAT"].to_numpy()

	# prepare the labelisation
	to_keep=list(feats)+list(categorical)
	train = train[to_keep]
	test = test[to_keep]
	train, test = labelisationAdversarialValidation(train, test)

	# adversarial validation with model
	result_model = adversarialValidationModel(train.append(test), TARGET_ADVERSARIAL, tps_func)

	# combine scores
	# variable with high p_value does not have score with model so we fill with 1
	result = pd.merge(result_ks, result_model, on="FEAT", how="left").fillna(1)
	result["SCORE_GLOBAL"] = (result["P_VALUE"] + result["SCORE"])/2
	# feats with p_value at 0 will have global score at 0, whatever the score of the model
	result["IS_ZERO"] = np.where(result["P_VALUE"]==0, 0, 1)
	result["SCORE_GLOBAL"] *= result["IS_ZERO"]
	result = result.drop(labels="IS_ZERO", axis=1)

	suggestion = result[result["SCORE_GLOBAL"]>=threshold_ks]["FEAT"].to_list()

	return result, suggestion

