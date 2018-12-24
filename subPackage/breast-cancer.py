# -*- coding: utf-8 -*-
"""
Author: @0xskywalker(Skywalker)
Created on Fri Oct 26 09:06:30 2018

"""

import subPackage
from algorithms import Implementations

#1. Breast Cancer Wiconsin.data
#preprocessing
h = ['ID Number', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
    'Compactness', 'Concave points', 'Symmetry', 'Fractal dimension']
load_cancer = subPackage.pd.read_csv('breast-cancer-wisconsin.data', header=None, names=h)
load_cancer = subPackage.shuffle(load_cancer)
df_cancer = load_cancer.replace('?', 0)

cancer_target = df_cancer['Fractal dimension']
cancer_data = df_cancer.iloc[:,0:9]

#visualization
load_cancer.plot(kind='hist')

#5 Fold
k = 5 
#10 times
n = 10
Implementations(k, n).classification(cancer_data, cancer_target)

