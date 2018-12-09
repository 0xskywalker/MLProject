# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 00:23:38 2018

"""

import subPackage
from algorithms import Implementations

#4. Car Evaluation.data

#preprocessing
headc = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
load_car = subPackage.pd.read_csv('car.data', header=None, names=headc)


load_car['buying'] = load_car['buying'].replace({'vhigh': 1, 'high': 2, 'med': 3, 'low':4})
load_car['maint'] = load_car['maint'].replace({'vhigh': 1, 'high': 2, 'med': 3, 'low': 4})
load_car['doors'] = load_car['doors'].replace({'more': 6, '5more': 6})
load_car['persons'] = load_car['persons'].replace('more', 6)
load_car['lug_boot'] = load_car['lug_boot'].replace({'small': 1, 'med': 2, 'big': 3})
load_car['safety'] = load_car['safety'].replace({'low':1, 'med': 2, 'high': 3})
load_car['class'] = load_car['class'].replace({'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4})


car_target = load_car['class']
car_data = load_car.iloc[:,0:6]

#visualization
load_car.plot(kind='hist')

#5 fold
k = 5
#10 times
n = 10

Implementations(k, n).classification(car_data, car_target)





