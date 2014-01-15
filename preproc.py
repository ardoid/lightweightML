#---------------------------------------------------------------------
# Data preprocessing
# Author : Ardo
#---------------------------------------------------------------------

import csv
pos = []
with open('dataset.csv', 'rb') as f:
	reader = csv.reader(f,delimiter=';')
	heading = reader.next()
	max = 10
	count = 0
	print heading
	for row in reader:
		print row
		count = count + 1
		if count==10:
			break