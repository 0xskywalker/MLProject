# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:29:45 2018

"""

import subPackage
from algorithms import Implementations

#3. Ecoli.data
subPackage.pd.set_option('display.max_rows', 1000)
subPackage.pd.set_option('display.max_columns', 100)
subPackage.pd.set_option('display.max_colwidth', 100)
subPackage.pd.set_option('display.width', None)
subPackage.pd.option_context('display.colheader_justify', 'right')

load_ecoli = subPackage.pd.read_csv('ecoli.data', header=None)
a= load_ecoli.to_csv('ecoli.csv')

file = open('ecoli.csv', 'r')
ac = subPackage.csv.reader(file)
nload_ecoli = [['Sequence name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2']]
for row in ac:
    data = row[1]
    data = data.split()
    data = data[1:9]
    if data == []:
        pass
    elif data[7] == 'cp':
        data[7] = 1
        #nload_ecoli.append(data)
    elif data[7] == 'im':
        data[7] = 2
    elif data[7] == 'pp':
        data[7] = 3
    elif data[7] == 'imU':
        data[7] = 4
    elif data[7] == 'om':
        data[7] = 5
    elif data[7] == 'omL':
        data[7] = 6
    elif data[7] == 'imL':
        data[7] = 7
    elif data[7] == 'imS':
        data[7] = 8
    for i in range(len(data)):
        data[i] = float(data[i])
    nload_ecoli.append(data)

nload_ecoli.remove(['Sequence name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'])
nload_ecoli.remove([])
nload_ecoli = subPackage.np.array(nload_ecoli)


ecoli_target = nload_ecoli[:, -1]
ecoli_data = nload_ecoli[:, :-1]

#5 fold
k = 5
#10 times
n = 10

Implementations(k, n).classification(ecoli_data, ecoli_target)



