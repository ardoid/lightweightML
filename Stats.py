#--------------------------------------------------------------#
# Statistic file class										   #
# Provide all the stats of the input data 					   #
# Author : Ardo												   #
#--------------------------------------------------------------#

import os.path
import json

class Stats:
	statsFile    = 'stats.txt'
	inputFile    = 'dataset.csv'
	datatypeFile = 'datatype.csv'

	minVal=[]
	maxVal=[]
	typeOfFeats=[]
	classLabel=[]
	totalData=0
	numOfFeat=0

	def __init__(self):
		self.loadTypes()
		self.loadStats()

	def loadTypes(self):
		dt = open(self.datatypeFile,'rb')
		self.typeOfFeats=dt.readline().split(';')
		dt.close()

	def loadStats(self):
		stats = open(self.statsFile,'rb')
		fileLen=len(stats.readlines())
		stats = open(self.statsFile,'rb')
		stats.readline()
		stats.readline()
		for i in range(0,fileLen):
			line=stats.readline()
			buf=line.split(' ')
			if buf[0]=='Total:':
				self.totalData=int(buf[1])
				break
			self.classLabel.append(line.strip().split('\t\t\t'))
		stats.readline()
		stats.readline()
		self.minVal=json.loads(stats.readline())
		stats.readline()
		self.maxVal=json.loads(stats.readline())
		buf=stats.readline().split(' ')
		if buf[0]=='Features:':
			self.numOfFeat=int(buf[1])
		stats.close()
