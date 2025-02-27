from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import re

def removeSymbols(sentence):
	sentence = re.sub(r'[^\w]', ' ', sentence) # \w is number or letter or _  ^\w all the chacaters without number and letter and _  r is represent is regx
	sentence = sentence.replace("_", "")
	sentence = ' '.join(sentence.split())
	return sentence

ppt = open("microsoftPPTX.txt", "r", encoding = 'utf-8')
processed = open("processed_ppt.dat", "w", encoding = 'utf-8')
text = ppt.readlines()
print(text)
for line in text:
	processedline = removeSymbols(line)
	if len(processedline) != 0:
		processedline = processedline +'\n'
		processed.write(processedline)
ppt.close()
processed.close()
newfile = open("processed_ppt.dat", "r")
print(newfile.readlines())
#processed = open("processed_ppt.dat", "w")
