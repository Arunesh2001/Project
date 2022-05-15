# Importing the libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
import scipy.stats as stats 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # To supress warnings
 # set the background for the graphs
from scipy.stats import skew
plt.style.use('ggplot')
#import missingno as msno # to get visualization on missing values
from sklearn.model_selection import train_test_split # Sklearn package's randomized data splitting function
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_colwidth',400)
pd.set_option('display.float_format', lambda x: '%.5f' % x) # To supress numerical display in scientific notations
import statsmodels.api as sm
# Importing LabelEncoder from Sklearn
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder
print("Load Libraries- Done")

import pandas as pd
df1 = pd.read_csv(r'D:\intern\tc20171021.csv',on_bad_lines='skip')
df2 = pd.read_csv(r'D:\intern\true_car_listings.csv')
df_full = pd.concat([df1, df2], ignore_index=True)
df_full
df_full.info()
df_full=df_full.copy()
print(f'There are {df_full.shape[0]} rows and {df_full.shape[1]} columns')
df_full.head(5)
df_full.tail(5)
#get the size of dataframe
print ("Rows     : " , df_full.shape[0])  #get number of rows/observations
print ("Columns  : " , df_full.shape[1]) #get number of columns
print ("#"*40,"\n","Features : \n\n", df_full.columns.tolist()) #get name of columns/features
print ("#"*40,"\nMissing values :\n\n", df_full.isnull().sum().sort_values(ascending=False))
print( "#"*40,"\nPercent of missing :\n\n", round(df_full.isna().sum() / df_full.isna().count() * 100, 2)) # looking at columns with most Missing Values
print ("#"*40,"\nUnique values :  \n\n", df_full.nunique())  
#  count of unique values
import missingno as msno
msno.bar(df_full)
# Making a list of all categorical variables
cat_col = [
    "City",
    "State",
    "Vin",
    "Make",
    "Model"    
]
# Printing number of count of each unique value in each column
for column in cat_col:
    print(df_full[column].value_counts())
    print("#" * 40)
#Data Preprocessing
num=['Price','Year','Mileage']
df_full[num].sample(20)

df_full[df_full.Price.isnull()==True]
df_full[df_full.Year.isnull()==True]
df_full[df_full.Mileage.isnull()==True]
df_full[num].nunique()
df_full.describe().T
plt.style.use('ggplot')
#select all quantitative columns for checking the spread
numeric_columns = df_full.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(20,25))

for i, variable in enumerate(numeric_columns):
                     plt.subplot(10,3,i+1)
                       
                     sns.distplot(df_full[variable],kde=False,color='blue')
                     plt.tight_layout()
                     plt.title(variable)
numeric_columns= numeric_columns = df_full.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(13,17))

for i, variable in enumerate(numeric_columns):
                     plt.subplot(5,2,i+1)
                     sns.scatterplot(x=df_full[variable],y=df_full['Price']).set(title='Price vs '+ variable)
                     #plt.xticks(rotation=90)
                     plt.tight_layout()
df_full.isnull().sum()
# counting the number of missing values per row
num_missing = df_full.isnull().sum(axis=1)
num_missing.value_counts()
#Investigating how many missing values per row are there for each variable
for n in num_missing.value_counts().sort_index().index:
    if n > 0:
        print("*" *30,f'\nFor the rows with exactly {n} missing values, NAs are found in:')
        n_miss_per_col = df_full[num_missing == n].isnull().sum()
        print(n_miss_per_col[n_miss_per_col > 0])
        print('\n\n')
col=['Make','Model','Mileage']
df_full[col].isnull().sum()
# check distrubution if skewed. If distrubution is skewed , it is advice to use log transform
cols_to_log = df_full.select_dtypes(include=np.number).columns.tolist()
for colname in cols_to_log:
    sns.distplot(df_full[colname], kde=True)
    plt.show()
def Perform_log_transform(df,col_log):
    """#Perform Log Transformation of dataframe , and list of columns """
    for colname in col_log:
        df_full[colname + '_log'] = np.log(df_full[colname])
    #df.drop(col_log, axis=1, inplace=True)
    df_full.info()
Perform_log_transform(df_full,['Mileage','Price'])
df_full.Model.unique()
X = df_full.drop(["Price"], axis=1)
y = df_full[["Price"]]

y

def encode_cat_vars(x):
    x = pd.get_dummies(x, columns=["State"], drop_first=True)
    return x
#Dummy variable creation is done before spliting the data , so all the different categories are covered
#create dummy variable
X = encode_cat_vars(X)


# Importing LabelEncoder from Sklearn
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder
 
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
label_city = le.fit_transform(df_full['City'])
 
# printing label
label_city


# removing the column 'Purchased' from df
# as it is of no use now.
df_full.drop("City", axis=1, inplace=True)
 
# Appending the array to our dataFrame
# with column name 'Purchased'
df_full["City"] = label_city
 
# printing Dataframe
df_full


 
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
label_state = le.fit_transform(df_full['State'])
 
# printing label
label_state


# removing the column 'Purchased' from df
# as it is of no use now.
df_full.drop("State", axis=1, inplace=True)
 
# Appending the array to our dataFrame
# with column name 'Purchased'
df_full["State"] = label_state
 
# printing Dataframe
df_full
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
label_vin = le.fit_transform(df_full['Vin'])
 
# printing label
label_vin
# removing the column 'Purchased' from df
# as it is of no use now.
df_full.drop("Vin", axis=1, inplace=True)
 
# Appending the array to our dataFrame
# with column name 'Purchased'
df_full["Vin"] = label_vin
 
# printing Dataframe
df_full
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
label_make = le.fit_transform(df_full['Make'])
 
# printing label
label_make
# removing the column 'Purchased' from df
# as it is of no use now.
df_full.drop("Make", axis=1, inplace=True)
 
# Appending the array to our dataFrame
# with column name 'Purchased'
df_full["Make"] = label_make
 
# printing Dataframe
df_full
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
label_model = le.fit_transform(df_full['Model'])
 
# printing label
label_model
# removing the column 'Purchased' from df
# as it is of no use now.
df_full.drop("Model", axis=1, inplace=True)
 
# Appending the array to our dataFrame
# with column name 'Purchased'
df_full["Model"] = label_model
 
# printing Dataframe
df_full
print(df_full.head())
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier

# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

 #Feature Engineering
q1 = df_full['Mileage'].quantile(0.20)
q3 = df_full['Mileage'].quantile(0.70)
def amount_of_mileage(mileage):
   
    if mileage <= q1:
        return "Low"
    elif (mileage > q1) and (mileage < q3 ):
        return "Medium"
    else:
        return "High"
df_full["level"] = df_full["Mileage"].apply(amount_of_mileage)
df_full
plt.figure(figsize=(9,9))
sns.set_theme(style="ticks")
ax = sns.countplot(x="level", data=df_full)
plt.show()
#b) Old and New cars by year

#Year Column:Weconsider mostly car manufactured between the year of 2000 and 2018
# Cars that manufacutred before 2000
df_full_old_cars = df_full[df_full["Year"] < 2000]
# Cars that manufacutred after 2000
df_full_new_cars = df_full[df_full["Year"] >=  2000]
df_full_old_cars.count()
df_full_new_cars.count()
#Difference between prices of old and new cars -
df_full_old_cars.Price.mean()
df_full_new_cars.Price.mean()
increase = round((df_full_new_cars.Price.mean() /df_full_old_cars.Price.mean()*100),2)
print("Average of new cars is " + str(increase) + " % higher than old cars")
df_full.City.unique()
#Importance of cities with respect to sales:
def importance_of_cities(City):
    important_cities = ["New York", "Los Angeles", "Chicago","Washington", "Houston", "Phoenix", 
                        "San Antonio", "San Diego", "San Francisco"]
    Medium_cities = ["Austin", "Charlotte", "San Jose",
                     "Indianapolis", "Seattle", "Denver", "Houston"]
    important_cities = [c.upper() for c in important_cities]
    Midium_cities = [c.upper() for c in Medium_cities]
    if City in important_cities:
        return "High"
    elif City in Midium_cities:
        return "Medium"
    else:
        return "Low"

df_full["Importance_of_cities"] = df_full["City"].apply(importance_of_cities)
#Plotting Usage Level:
plt.figure(figsize=(7,7))
sns.set_theme(style="ticks")
plt.title("Importance of Cities", fontsize = 10)
sns.countplot(x= df_full["Importance_of_cities"])
plt.show()
df = df_full.drop(['level','Importance_of_cities','State'],axis=1)
df.head()
from statsmodels.stats.outliers_influence import variance_inflation_factor
# the independent variables set
variables = df[['Make', 'Model', 'City','Mileage','Year']]
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = variables.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(variables.values, i)
                          for i in range(len(variables.columns))]
  
print(vif_data)

# the independent variables set
variables1 = df[['Mileage', 'City', 'Make', 'Model']]
  
 # VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = variables1.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(variables1.values, i)
                          for i in range(len(variables1.columns))]
  
print(vif_data)
# drop 'year' column
df.drop(columns = ['Year','Mileage_log','Price_log','Vin','Id'], inplace = True)
df.head()
# split data
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
x = df.drop(['Price'],axis=1)
y= df[['Price']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 101)
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostRegressor
from numpy import array
estimator = AdaBoostRegressor(random_state=0, n_estimators=100)
selector = RFE(estimator, n_features_to_select=8, step=1)
selector = selector.fit(x, y)
filter = selector.support_
ranking = selector.ranking_

print("Mask data: ", filter)
print("Ranking: ", ranking)
selector.estimator_.feature_importances_
features = array(x.columns)
print("All features:")
print(features)

print("Selected features:")
print(features[filter])
#Splitting the dependant and independant features
Y = df.Price  
X = df[['Mileage','City', 'Make', 'Model']]
#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Fitting the Gradient Boosting Regressor Model to the dataset#
from sklearn.ensemble import GradientBoostingRegressor
GB_Reg = GradientBoostingRegressor()
# Fitting the regressor to our training set
GB_Reg.fit(X_train, y_train)
#Predicting the x_test
Y_pred = GB_Reg.predict(X_test)
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
