#--------------------------------------------------------------
# Data preprocessing
# Author : Ardo
#--------------------------------------------------------------

import csv
import json
import os.path

statsFile = 'stats.txt'
inputFile = 'dataset.csv'

def splitDataByClass(inputFile):
	listOfClass = []
	numOfClass  = []
	outputFile  = []
	with open(inputFile, 'rb') as f:
		reader = csv.reader(f,delimiter=';')
		heading = reader.next()
		# count = 0
		print heading
		for row in reader:
			# print row
			if row[-1] not in listOfClass:
				listOfClass.append(row[-1])
				numOfClass.append(1)
				fName = 'o'+str(len(numOfClass))+'.txt'
				output = open(fName,'w')
				outputFile.append(output)
			else:
				indexOfClass = listOfClass.index(row[-1])
				outputFile[indexOfClass].write(json.dumps(row))
				outputFile[indexOfClass].write('\n')
				numOfClass[indexOfClass]=numOfClass[indexOfClass]+1
			# count = count + 1
			# if count==1000:
			# 	break
		output = open(statsFile,'w')
		output.write('Class\t\t\t Size\n')
		output.write('----------------\n')
		idx = 0
		sums = 0
		for cl in listOfClass:
			output.write(cl+'\t\t\t'+str(numOfClass[idx])+'\n')
			sums = sums + numOfClass[idx]
			idx = idx+1
		output.write('Total: '+str(sums))

def splitData10fcv(inputFile):
	print 'a'

#start of the preprocessing program
if not (os.path.isfile(statsFile)):
	splitDataByClass(inputFile)
else:
	print 'process'









