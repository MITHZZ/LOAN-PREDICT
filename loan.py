import pandas as pd
import numpy as np
import matplotlib as plt
""
''

#Data Munging

df = pd.read_csv("train.csv")
#df['ApplicantIncome'].hist(bins=50)
#df.boxplot(column='ApplicantIncome',by='Education')
df['LoanAmount'].hist(bins=10)

temp1=df['Credit_History'].value_counts(ascending=True)
temp2=df.pivot_table(values='Loan_Status',index=['Credit_History'],
               aggfunc=lambda x : x.map({'Y':1,'N':0}).mean())

print ('Frequency Table for Credit History:' )
print (temp1)

print ('\nProbility of getting loan for each Credit History class:' )
print (temp2)

##we can observe that we get a similar pivot_table. This can be plotted as a bar chart using the “matplotlib” library with following code:

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1=fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

#checking for null values
print("Null values :")
print(df.apply(lambda x: sum(x.isnull()),axis=0))

#filling the missing loan amounts 
df['Self_Employed'].fillna('No',inplace=True)
table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

''
df['Gender'].fillna('Male',inplace=True)
df['Married'].fillna('Yes',inplace=True)
df['Dependents'].fillna(0,inplace=True)
df['Self_Employed'].fillna('No',inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['Credit_History'].fillna(1,inplace=True) 

#outliers are converted to  a log transformation to nullify their effect:
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 

#Building a Predictive Model 
#onvert all our categorical variables into numeric by encoding the categories

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
var_mod =['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
df.dtypes

#var_mod = [int(i) for i in var_mod]
 #,'Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']

#df[i] = le.fit_transform(df[i])
le = LabelEncoder()
for i in var_mod:
 df[i] = le.fit_transform(df[i].astype('str'))

 

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
 #Fit the model:
 model.fit(data[predictors],data[outcome])
 #Make predictions on training set:
 predictions = model.predict(data[predictors])
 #Print accuracy
 accuracy = metrics.accuracy_score(predictions,data[outcome])
 print( "Accuracy : %s" % "{0:.3%}".format(accuracy))
 
 #Perform k-fold cross-validation with 5 folds
 kf = KFold(data.shape[0], n_folds=5)
 error = []
 for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
 print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
 model.fit(data[predictors],data[outcome])
 
#Logistic Regression
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, df,predictor_var,outcome_var)
#We can try different combination of variables:


#Decision Tree
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model, df,predictor_var,outcome_var)

#Random Forest
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, df,predictor_var,outcome_var)


#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, df,predictor_var,outcome_var)


