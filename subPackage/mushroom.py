# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:49:42 2018

"""

import subPackage
from algorithms import Implementations

#5. Mushroom.data
#preprocessing
h = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attc',
    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
    'stalk-surface-ar', 'stalk-surface-br', 'stalk-color-ar', 'stalk-color-br',
    'veil-type', 'veil-color', 'ring-num', 'ring-type', 'spore-print-col',
    'population', 'habitat']
load_mushroom = subPackage.pd.read_csv('mushroom.data', header=None, names=h)

f0 = {'e':0, 'p':1}
load_mushroom['target'] = load_mushroom['target'].replace(f0)
f1 = {'b':1, 'c':2, 'x':3, 'f':4, 'k':5, 's':6}
load_mushroom['cap-shape'] = load_mushroom['cap-shape'].replace(f1)
f2 = {'f':1, 'g':2, 'y':3, 's':4}
load_mushroom['cap-surface'] = load_mushroom['cap-surface'].replace(f2)
f3 = {'n':1, 'b':2, 'c':3, 'g':4, 'r':5, 'p':6, 'u':7, 'e':6, 'w':7, 'y':8}
load_mushroom['cap-color'] = load_mushroom['cap-color'].replace(f3)
f4 = {'t':1, 'f':2}
load_mushroom['bruises'] = load_mushroom['bruises'].replace(f4)
f5 = {'a':1, 'l':2, 'c':3, 'y':4, 'f':5, 'm':6, 'n':7, 'p':8, 's':9}
load_mushroom['odor'] = load_mushroom['odor'].replace(f5)
f6 = {'a':1, 'd':2, 'f':3, 'n':4}
load_mushroom['gill-attc'] = load_mushroom['gill-attc'].replace(f6)
f7 = {'c':1, 'w':2, 'd':3}
load_mushroom['gill-spacing'] = load_mushroom['gill-spacing'].replace(f7)
f8 = {'b':1, 'n':2}
load_mushroom['gill-size'] = load_mushroom['gill-size'].replace(f8)
f9 = {'k':1, 'n':2, 'b':3, 'h':4, 'g':5, 'r':6, 'o':7, 'p':8, 'u':9, 'e':10, 'w':11, 'y':12}
load_mushroom['gill-color'] = load_mushroom['gill-color'].replace(f9)
f10 = {'e':1, 't':2}
load_mushroom['stalk-shape'] = load_mushroom['stalk-shape'].replace(f10)
f11 = {'b':1, 'c':2, 'u':3, 'e':4, 'z':5, 'r':6, '?':0}
load_mushroom['stalk-root'] = load_mushroom['stalk-root'].replace(f11)
f12 = {'f':1, 'y':2, 'k':3, 's':4}
load_mushroom['stalk-surface-ar'] = load_mushroom['stalk-surface-ar'].replace(f12)
load_mushroom['stalk-surface-br'] = load_mushroom['stalk-surface-br'].replace(f12)
f13 = {'n':1, 'b':2, 'c':3, 'g':4, 'o':5, 'p':6, 'e':7, 'w':8, 'y':9}
load_mushroom['stalk-color-ar'] = load_mushroom['stalk-color-ar'].replace(f13)
load_mushroom['stalk-color-br'] = load_mushroom['stalk-color-br'].replace(f13)
f14 = {'p':1, 'u':2}
load_mushroom['veil-type'] = load_mushroom['veil-type'].replace(f14)
f15 = {'n':1, 'o':2, 'w':3, 'y':4}
load_mushroom['veil-color'] = load_mushroom['veil-color'].replace(f15)
f16 = {'n':1, 'o':2, 't':3}
load_mushroom['ring-num'] = load_mushroom['ring-num'].replace(f16)
f17 = {'c':1, 'e':2, 'f':3, 'l':4, 'n':5, 'p':6, 's':7, 'z':8}
load_mushroom['ring-type'] = load_mushroom['ring-type'].replace(f17)
f18 = {'k':1, 'n':2, 'b':3, 'h':4, 'r':5, 'o':6, 'u':7, 'w':8, 'y':9}
load_mushroom['spore-print-col'] = load_mushroom['spore-print-col'].replace(f18)
f19 = {'a':1, 'c':2, 'n':3, 's':4, 'v':5, 'y':6}
load_mushroom['population'] = load_mushroom['population'].replace(f19)
f20 = {'g':1, 'l':2, 'm':3, 'p':4, 'u':5, 'w':6, 'd':7}
load_mushroom['habitat'] = load_mushroom['habitat'].replace(f20)

mushroom_target = load_mushroom['target']
mushroom_data = load_mushroom.iloc[:,1:23]

#visualization
load_mushroom.plot(kind='hist')


#5 fold
k = 5
#10 times
n = 10
Implementations(k, n).classification(mushroom_data, mushroom_target)



