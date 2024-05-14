#Importing the packages
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Reading test_med.csv dataset
os.getcwd()
os.chdir('C:\\Users\\SHAIFALI PATWAL\\Desktop\\Github Projects')
d1 = pd.read_csv('test_med.csv')

# first 5 rows of d1
d1.head()

# Subsetting d1 
d2=d1[['ID', 'SBP', 'DBP', 'AGE', 'BMI', 'RACE']]

# first 5 rows after subsetting
d2.head()

# Deleting missing values 
d3=d2.dropna()

# Summary Statistics
d3.describe()

# Data Exploration
# Histograms of variables
d3.hist(figsize=(10, 8))
plt.show()


# Scatter plots of variables against each other
pd.plotting.scatter_matrix(d3, figsize=(12, 10))
plt.show()


# Fitting a linear model without random intercept
ri_lm = smf.ols("DBP ~ AGE + BMI + RACE", data=d3).fit()
print(ri_lm.summary())


# Fitting a linear mixed model with ID as a random intercept and AGE, BMI and RACE as covariates
ri_mixed = smf.mixedlm('DBP ~ AGE + BMI + RACE', data=d3, groups=d3['ID']).fit()
print(ri_mixed.summary())


# Computing the confidence intervals
conf_int = ri_mixed.conf_int()
print(conf_int)

# Computing the random effects
random_effects = ri_mixed.random_effects
print(pd.DataFrame(random_effects).head())

# Computing various model predictions
predict_interval = ri_mixed.predict()
print(predict_interval)



