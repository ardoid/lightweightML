#--------------------------------------------------------------#
# Data processing											   #
# Machine learning and hyperparameters optimization 		   #
# Author : Ardo												   #
#--------------------------------------------------------------#

import json
import os.path
import sys
import random
import Stats
from numpy import *

inputFile    = 'dataset.csv'

def loadData(method):
	stats = Stats.Stats()
	
	if method=='KFCV':
		loadKFCV(K)
	elif method=='s':
		loadMethod()
	else:
		print 'Error: Unknown method'

def loadKFCV(K):
	pass

def mlKnn(inputData,K):
	pass
	
loadData('KFCV')
print stats.minVal
print stats.typeOfClass
