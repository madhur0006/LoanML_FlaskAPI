# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:32:47 2020

@author: Madhur
"""

# Importing the libraries
import pandas as pd

# if numpy is not installed already : pip3 install numpy
import numpy as np

# matplotlib: used to plot graphs
import matplotlib
import scipy.stats as stats
matplotlib.use('nbagg')
import matplotlib.pylab as plt
import seaborn as sns#Plots
#from matplotlib import rcParams #Size of plots  

#import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")



df_data=pd.read_excel("sample data2_Translated.xlsx")

# Seprating Numeric and Categorica columns

numeric_var_names=[key for key in dict(df_data.dtypes) if dict(df_data.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
cat_var_names=[key for key in dict(df_data.dtypes) if dict(df_data.dtypes)[key] in ['object']]

df_data_num=df_data[numeric_var_names]
df_data_cat=df_data[cat_var_names]

# Check Stats about the data
 
# Num Data
def num_var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_summary=df_data_num.apply(lambda x: num_var_summary(x)).T
print(num_summary)

# Cat Data

def cat_var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.value_counts()], 
                  index=['N', 'NMISS', 'ColumnsNames'])

cat_summary=df_data_cat.apply(lambda x: cat_var_summary(x)).T

print(cat_summary)

# Handling Outlier:

def outlier_capping(x):
    x = x.clip_upper(x.quantile(0.99))
    x = x.clip_lower(x.quantile(0.01))
    return x

df_data_num=df_data_num.apply(lambda x: outlier_capping(x))


# Check percentage of Missing Values:

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

print('Percentage of Missing value in Each Columns:')    
print(missing_values_table(df_data))

# Impute Missing value in NUM data

#Handling missings - Method2
def Missing_imputation(x):
    x = x.fillna(x.median())
    return x

df_data_num=df_data_num.apply(lambda x: Missing_imputation(x))

#Check T-Stats in NUm data columns:

tstats_df = pd.DataFrame()
for num_variable in df_data_num:
    tstats = stats.ttest_ind(df_data_num[df_data_num['OK?']==1][num_variable],df_data_num[df_data_num['OK?']==0][num_variable])
    temp = pd.DataFrame([num_variable, tstats[0], tstats[1]]).T
    temp.columns = ['Variable Name', 'T-Statistic', 'P-Value']
    tstats_df = pd.concat([tstats_df, temp], axis=0, ignore_index=True)


# Handle Categorical Data

# Impute missing value in categorical data

def missing_value_imputation(df):

    df["Gender"].fillna(df["Gender"].mode()[0],inplace=True)
    df["Birth City"].fillna(df["Birth City"].mode()[0],inplace=True)
    df["as_us_contact"].fillna(df["as_us_contact"].mode()[0],inplace=True)
    df["purpose_credit"].fillna(df["purpose_credit"].mode()[0],inplace=True)
    df.drop(['position_work','position','company.1','company'],axis=1,inplace=True)
    
    return df
missing_value_imputation(df_data_cat)

# Encode Categiorical Data:

def covert_categorical_features(df):
    
    fe1 =df.groupby('Birth City').size()/len(df)
    df.loc[:,'Birth City_Freq']=df['Birth City'].map(fe1)
        
    fe2 =df.groupby('as_us_contact').size()/len(df)
    df.loc[:,'as_us_contact_Freq']=df['as_us_contact'].map(fe2)
        
    fe3 =df.groupby('purpose_credit').size()/len(df)
    df.loc[:,'purpose_credit_Freq']=df['purpose_credit'].map(fe3)
        
    df['Gender_Male_Female_encode']=np.where((df.Gender=='M'),1,0)
    
    df.to_csv ('categorical_encode.csv', index = False, header=True)
    
    df.drop(['Birth City','as_us_contact','purpose_credit','Gender'],axis=1,inplace=True)
    
    return df

covert_categorical_features(df_data_cat)


# Join both Num and Cat columns:

df_final = pd.concat([df_data_num, df_data_cat], axis=1)


# Check Corelation in data in data:

correlation = df_final.corr()
plt.subplots(figsize=(30,10))
sns.heatmap( correlation, square=True, annot=True, fmt=".1f" )
plt.show();

# Check Correlation and delete the columns with high correlation :


columns = np.full((correlation.shape[0],), True, dtype=bool)
for i in range(correlation.shape[0]):
    for j in range(i+1, correlation.shape[0]):
        if correlation.iloc[i,j] >= 0.8:
            if columns[i]:
                columns[i] = False
selected_columns = df_final.columns[columns]
df_final = df_final[selected_columns]


#Remove Duplicate data
df_final.drop(['account agreement payment'],axis=1,inplace=True)
df_final=df_final.loc[~df_final.duplicated(),:]
df_final.to_csv('Final_Data.csv')

# Seprating the data: X and Y

X_Features=df_final.columns.difference(['OK?'])

#X=df_final[X_Features]
#Y=df_final['OK?']

# Train test Split:

###################################################################################
# Training the model with Sample data

X_train, X_test, y_train, y_test = train_test_split(df_final[X_Features],
                                                   df_final['OK?'],
                                                   test_size = 0.05,
                                                   random_state = 142 )
 
 
 # As the data we show is implanced one so did oversampling
 
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 
   
 # import SMOTE module from imblearn library 
 # pip install imblearn (if you don't have imblearn in your system) 
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 
   
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
   
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))
 
 
 # Training the Model
lr = LogisticRegression(random_state=2)
 
params = {
     "C": [10 ** x for x in range(-5, 5)],
     "penalty": ['l1', 'l2'],
     "solver": ["newton-cg", "lbfgs", "sag", "saga"]
}
 
grid = GridSearchCV(lr, param_grid=params, scoring="f1_weighted", cv=3, verbose=10)
grid.fit(X_train_res, y_train_res)
 
print("The Best Weighted F1-Score: ", grid.best_score_)
print("The Best Parameters are: ", grid.best_params_)
 
bst = grid.best_estimator_
pred_test = bst.predict(X_test)
bst = grid.best_estimator_
pred_test = bst.predict(X_test)
 
roc=roc_auc_score(y_test, pred_test)
acc = accuracy_score(y_test, pred_test)
prec = precision_score(y_test, pred_test)
rec = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
 
results = pd.DataFrame([['Logistic Regression', acc,prec,rec, f1,roc]],
                columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
 
print ('Model Accuracy')
print(results)
 
print('Confusioin Matrix')
print(confusion_matrix(y_test, pred_test))
#####################################################################




# Train he model with Full data:


X=df_final[X_Features]
Y=df_final['OK?']

print("Before OverSampling, counts of label '1': {}".format(sum(Y == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(Y == 0))) 
  
# import SMOTE module from imblearn library 
# pip install imblearn (if you don't have imblearn in your system) 
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_res, Y_res = sm.fit_sample(X, Y.ravel()) 
  
print('After OverSampling, the shape of train_X: {}'.format(X_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(Y_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(Y_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(Y_res == 0)))



#TRAIN THE MODEL:

# Training the Model
lr = LogisticRegression(random_state=2)

params = {
    "C": [10 ** x for x in range(-5, 5)],
    "penalty": ['l1', 'l2'],
    "solver": ["newton-cg", "lbfgs", "sag", "saga"]
}

Model_Full_Data = GridSearchCV(lr, param_grid=params, scoring="f1_weighted", cv=3, verbose=10)
Model_Full_Data.fit(X_res, Y_res)

print("The Best Weighted F1-Score: ", Model_Full_Data.best_score_)
print("The Best Parameters are: ", Model_Full_Data.best_params_)

X_res.to_csv('Final_Data_X_Level.csv')


# Saving model to disk
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(Model_Full_Data, file)

# Loading model to compare the results
#model = pickle.load(open('pickle_model.pkl','rb'))
#prediction_arr=model.predict_proba([[3,29,30,0.004168597,1,1,1,0,0,1,0,0,0.792033349,1,0,34,3,4200,0.623899954,22000,10
#]])
#prediction=np.array_split(prediction_arr,2)
#print(prediction[0][0][0])
#print(prediction[0][0][1])

#output_0 = prediction[0][0][0] *100
#output_1 = prediction[0][0][1] *100
#print('Percentage of Customer being Zero i.e Default Loan {} % and Percentage of Customer being Zero i.e Default Loan $ {} %'.format(output_0,output_1))

