import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.covariance as cv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import statsmodels as sm

df = pd.read_csv('../data/EstudioCrediticio_TrainP.csv')

print('TOTAL ENTRIES: ' + str(len(df.index)))
for column in df.columns:
    print(column + ': ' + str(df[column].isna().sum()) + ' N/A values')

print(df.head())

df2 = df.dropna()
df2 = pd.get_dummies(df2)

plt.matshow(df2.corr())
plt.show()
'''
df = df.dropna()



df2 = pd.get_dummies(df)

empirical = cv.EmpiricalCovariance()
skrunk = cv.ShrunkCovariance()
ledoitWolf = cv.LedoitWolf()
oas = cv.OAS()
graphicalLasso = cv.GraphicalLasso()
graphicalLassoCV = cv.GraphicalLassoCV()
minCovDet = cv.MinCovDet()


'''