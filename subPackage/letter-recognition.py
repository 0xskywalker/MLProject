# -*- coding: utf-8 -*-
"""
Author: @0xskywalker(Skywalker)
Created on Fri Oct 26 09:51:12 2018

"""

import subPackage
from algorithms import Implementations

#2. letter reognition.data
#preprocessing
h = ['Letter', 'x-box horizontal', 'y-box vertical', 'width of box',
    'height of box', 'total on pixel', 'x-bar mean', 'y-bar mean',
    'x2bar mean', 'y2bar mean', 'xybar mean', 'x2ybr mean', 'xy2bar mean',
    'x_edge mean', 'xegvy correlation', 'y-edge mean', 'yegvx correlation']
load_letter = subPackage.pd.read_csv('letter-recognition.data', header=None, names=h)

dic = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10, 'K':11, 'L':12, 'M':13, 'N':14, 'O':15, 'P':16, 'Q':17, 'R':18, 'S':19, 'T':20, 'U':21, 'V':22, 'W':23, 'X':24, 'Y':25, 'Z':26}
df_letter = load_letter.replace(dic)

letter_target = df_letter['Letter']
letter_data = df_letter.iloc[:,1:17]

letter_target = letter_target.astype(int)
letter_data = letter_data.astype(int)

#visualization
load_letter.plot(kind='hist')

#5 Fold
k = 5 
#10 times
n = 10
Implementations(k, n).classification(letter_data, letter_target)

