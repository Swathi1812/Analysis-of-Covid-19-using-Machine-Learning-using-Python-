# -*- coding: utf-8 -*-
"""
ANALYSING COVID-19 DATASETS USING MACHINE LEARNING

By: Swathi Mithanthaya

Batch 4: ML using Python 

OBJECTIVE:
   To analyse COVID-19 data of India and its states and UTs

DATASET:  
    1. 1_india_cases.csv : 
        daily total cases, new cases and deaths in India
    
    2. 7_statewise_testing_details.csv :
        State wise daily testing details
    
    3. statewise.csv :
        state wise daily cases, cured cases and deaths
"""

#Import pandas library to read_csv files

import pandas as pd

#Import all the datasets

cases_in_india = pd.read_csv("1_india_cases.csv")
cases_in_states = pd.read_csv("statewise.csv")
tests_in_states = pd.read_csv("7_statewise_testing_details.csv")


"""
OBJECTIVE:
    Analysing 1_india_cases.csv i.e the total cases in India

DATASET:
    This dataset is collected from 31-12-2019 to 09-08-2020
    It gives data about the new_cases, new_deaths which is then added to 
    total_cases and total_deaths respectively. Then these data is calculated
    for per million for the above given  dates. 
    There are no missing values and it is a continuous data

VARIABLES:
    Dependent : new_cases, new_deaths , total_cases , total_deaths
    Target : new_cases_per_million , new_deaths_per_million , 
             total_cases_per_million , total_deaths_per_million   
"""

#DATA CLEANING
cases_in_india.drop('date',axis='columns',inplace=True)

#DATA INTERPRETATION
#Let's analyse the data
#Checking size of the data i.e. no. of rows and columns 

cases_in_india.shape

#Checking the old few data and latest data in the dataset

cases_in_india.head()
cases_in_india.tail()

#Finding count,mean,standard deviation,minimum,maximum,interquartile range

cases_in_india.describe()

cases_in_india.info()

#DATA ANALYSIS

#SCATTER PLOT
#Inorder to see feature-feature relation
#Library: matplotlib
#Class: pyplot
#Func. : scatter
#import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
plt.scatter(cases_in_india['total_cases_per_million'],cases_in_india['total_deaths_per_million'],marker='.')


"""
OBJECTIVE:
    Analysing 7_statewise_testing_details.csv i.e the total tests in India

DATASET:
    This dataset is collected from 17-04-2019 to 05-08-2020
    It gives daily state wise data about the total tests and positive cases.
    There are lot of missing values in negative column and
    few in positive column and it is a continuous data

VARIABLES:
    Dependent : State, TotalSamples
    Target : Positive  
"""

#DATA CLEANING 

tests_in_states.drop('Date',axis='columns',inplace=True)

tests_in_states.drop('Negative',axis='columns',inplace=True)

tests_in_states.dropna(axis='rows',inplace=True)


#DATA INTERPRETATION
#Let's analyse the data
#Checking size of the data i.e. no. of rows and columns 

tests_in_states.shape

#Checking the old few data and latest data in the dataset

tests_in_states.head()
tests_in_states.tail()

#Finding count,mean,standard deviation,minimum,maximum,interquartile range

tests_in_states.describe()

tests_in_states.info()

#Check number of unique values in State/UT column

tests_in_states['State'].value_counts()


#DATA ANALYSIS

#SCATTER PLOT
#Inorder to see feature-feature relation
#Library: matplotlib
#Class: pyplot
#Func. : scatter
#import matplotlib.pyplot as plt

plt.scatter(tests_in_states['TotalSamples'],tests_in_states['State'],marker='.')

plt.scatter(tests_in_states['Positive'],tests_in_states['State'],marker='.')


#Considering that we need algorithm to predict the active cases state wise

#Create arrays
x1=tests_in_states.iloc[:,1:3].values
y1=tests_in_states.iloc[:,-1].values
#Normalising
#Standard Scaling
#MEAN = 0, SD=1, VALUES RANGE -2.56 to +2.56
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
x1=std.fit_transform(x1)
y1=y1.reshape(-1, 1)
y1=std.fit_transform(y1)

#Split the dataset
from sklearn.model_selection import train_test_split as tts
x1_train,x1_test,y1_train,y1_test=tts(x1,y1,train_size=0.8,random_state=72)

#ALGORITHM SELECTION
from sklearn.linear_model import LinearRegression
linreg1=LinearRegression()

#TRAINING
linreg1.fit(x1_train,y1_train)

#TESTING
#Accuracy
linregacc1=linreg1.score(x1_test,y1_test)
print(linregacc1)
#Predict
linregpred1=linreg1.predict(x1_test)
print(linregpred1)

#Scatter plot to see data distribution
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
plt.scatter(y1_test,linregpred1)


"""
OBJECTIVE:
    Analysing statewise.csv i.e the total cases in India

DATASET:
    This dataset is collected till 10-08-2020
    It gives state wise data about the total cases, cured cases and deaths. 
    There are no missing values and 
    it has a continuous data and categorical data

VARIABLES:
    Dependent : State
    Target : confirmed, Cured,Deaths   
"""

#DATA INTERPRETATION
#Let's analyse the data
#Checking size of the data i.e. no. of rows and columns 

cases_in_states.shape

#Checking the old few data and latest data in the dataset

cases_in_states.head()
cases_in_states.tail()

#Finding count,mean,standard deviation,minimum,maximum,interquartile range

cases_in_states.describe()

cases_in_states.info()

#DATA ANALYSIS

#BAR CHART
#Inorder to see feature-feature relation
#Library: matplotlib
#Class: pyplot
#Func. : bar
#import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
plt.bar("State", "Confirmed", data = cases_in_states, color = "orange")
plt.xlabel("State")
plt.xticks(rotation = 90)
plt.ylabel("Confirmed")
plt.title("Confirmed cases state wise")
plt.show()
plt.bar("State", "Cured", data = cases_in_states, color = "green")
plt.xlabel("State")
plt.xticks(rotation = 90)
plt.ylabel("Cured")
plt.title("Cured cases state wise")
plt.show()  

plt.bar("State", "Deaths", data = cases_in_states, color = "red")
plt.xlabel("State")
plt.xticks(rotation = 90)
plt.ylabel("Deaths")
plt.title("Deaths state wise")
plt.show()

cases_in_states['active']=cases_in_states['Confirmed']-cases_in_states['Cured']-cases_in_states['Deaths']

plt.bar("State", "active", data = cases_in_states, color = "blue")
plt.xlabel("State")
plt.xticks(rotation = 90)
plt.ylabel("active")
plt.title("active state wise")
plt.show()


#HEAT MAP
#Gives the details regarding variable correlation
import seaborn as sb
sb.heatmap(cases_in_states.corr(),annot=True)

#Considering that we need algorithm to predict the active cases state wise

#CREATE ARRAYS
x=cases_in_states.iloc[:,1:5].values
y=cases_in_states.iloc[:,4].values
#Normalising
#Standard Scaling
#MEAN = 0, SD=1, VALUES RANGE -2.56 to +2.56
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
x=std.fit_transform(x)
y=y.reshape(-1, 1)
y=std.fit_transform(y)

#Split the dataset
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8,random_state=72)

#ALGORITHM SELECTION
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()

#TRAINING
linreg.fit(x_train,y_train)

#TESTING
#Accuracy
linregacc=linreg.score(x_test,y_test)

#Predict
linregpred=linreg.predict(x_test)

#Scatter plot to see data distribution
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
plt.scatter(y_test,linregpred)

#SAVE MODEL (.pkl, .sav)
#JOBLIB

import joblib
joblib.dump(linreg, 'linearmodel.sav')
mymodel = joblib.load('linearmodel.sav')

joblib.dump(linreg1, 'linearmodel1.sav')
mymodel1 = joblib.load('linearmodel1.sav')








