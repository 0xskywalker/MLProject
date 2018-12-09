# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 08:58:58 2018

"""

import subPackage

class Implementations():
    def __init__(self, k, n):
        self.k = k
        self.n = n
    def regression(self, data, target):
        #10 times 5-fold cross validation
        rkf = subPackage.RepeatedKFold(n_splits=self.k, n_repeats=self.n, random_state=2)
        score = 'neg_mean_absolute_error'
        #Regression Algorithms
        ID3 = subPackage.DecisionTreeRegressor(criterion='entropy')
        Adaboost = subPackage.AdaBoostRegressor(n_estimators=100)
        RF = subPackage.RandomForestRegressor(n_estimators=100)
        NB = subPackage.GaussianNB()
        BG = subPackage.BaggingRegressor(base_estimator=NB, n_estimators=100)
        KNN_M = subPackage.KNeighborsRegressor()
        KNN_E = subPackage.KNeighborsRegressor(metric='euclidean')
        #cross validation and scoring
        models = [ID3, Adaboost, RF, NB, BG, KNN_M, KNN_E]
        names = ['id3', 'Adaboost on Tree stumps', 'Random Forest', 'NaiveBayes',
                'Bagging with NaiveBayes', 'KNearestNeighbor with Minkowski metric',
                'KNearestNeighbor with Euclidean metric']
        for model, name in zip(models, names):
            results = subPackage.cross_val_score(model, data, target, scoring=score, cv=rkf)
            print(name)
            print("Accuracy - Mean Absolute Error: ", subPackage.np.mean(results))
            print("Standard deviation: ", subPackage.np.std(results))
    def classification(self, data, target):
        #10 times stratified 5-fold cross validation
        rsf = subPackage.RepeatedStratifiedKFold(n_splits=self.k, n_repeats=self.n, random_state=2)
        score = 'accuracy'
        #Classification Algorithms
        ID3 = subPackage.DecisionTreeClassifier(criterion='entropy')
        Adaboost = subPackage.AdaBoostClassifier(n_estimators=100)
        RF = subPackage.RandomForestClassifier(n_estimators=100)
        NB = subPackage.GaussianNB()
        BG = subPackage.BaggingClassifier(base_estimator=NB, n_estimators=100)
        KNN_M = subPackage.KNeighborsClassifier(metric='minkowski')
        KNN_E = subPackage.KNeighborsClassifier(metric='euclidean')
        #cross validation and scoring
        models = [ID3, Adaboost, RF, NB, BG, KNN_M, KNN_E]
        names = ['id3', 'Adaboost on Tree stumps', 'Random Forest', 'NaiveBayes',
                'Bagging with NaiveBayes', 'KNearestNeighbor with Minkowski metric',
                'KNearestNeighbor with Euclidean metric']
        for model, name in zip(models, names):
            results = subPackage.cross_val_score(model, data, target, scoring=score, cv=rsf)
            print(name)
            print("Accuracy: ", subPackage.np.mean(results))
            print("Standard deviation: ", subPackage.np.std(results))
            
            
            
