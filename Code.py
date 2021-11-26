#Libraries to be used
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="white", color_codes=True)
#importing the data
df = pd.read_csv(r"D:\Data Analytics Portfolio Project\Loan Dataset.csv")
df.head(10) 
#To describe the entire dataset
df.describe()
#No. of Rows and Columns
df.shape
df.isnull().any()
df.isnull().sum()
df_loan = df.dropna()
df_loan.info()
df["Property_Area"].value_counts()
df['Dependents'].fillna(1,inplace=True)
df.info()
df['LoanAmount'].fillna(df.LoanAmount.mean(),inplace = True)
df.info()
df.head(10)
Value_Mapping = {'Yes' : 1, 'No' : 0}
df['Married_Section'] = df['Married'].map(Value_Mapping)
df.head(5)
Value_Mapping1 = {'Male' : 1, 'Female' : 0}
df['Gender_Section'] = df['Gender'].map(Value_Mapping1)
df.head(5)
df["Education"].unique()
df.head(5)
Value_Mapping2 = {'Graduate' : 1, 'Not Graduate' : 0}
df['Edu_Section'] = df['Education'].map(Value_Mapping2)
df.head(5)
df["Married_Section"].fillna(df.Married_Section.mean(),inplace=True) 

df["Gender_Section"].fillna(df.Gender_Section.mean(),inplace=True)

df["Loan_Amount_Term"].fillna(df.Loan_Amount_Term.mean(),inplace=True)

df["Credit_History"].fillna(df.Credit_History.mean(),inplace=True)
df.info()
Value_Mapping3 = {'Yes' : 1, 'No' : 0}
df['Employed_Section'] = df['Self_Employed'].map(Value_Mapping3)
df.head(5)
df["Employed_Section"].fillna(df.Employed_Section.mean(),inplace=True)
df.info()
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["Property_Section"] = lb_make.fit_transform(df["Property_Area"])
df.head(5)
Value_Mapping4 = {'Y' : 1, 'N' : 0}
df['Loan_Section'] = df['Loan_Status'].map(Value_Mapping4)
df.head(5)
# Using SCATTER REPRESENTATION
# Imported libraries... x-axis: loan_status, y_axis: Loan_Amount and representing in terms of Gender_Section

sns.FacetGrid(df,hue="Gender_Section",height=4) \
.map(plt.scatter,"Loan_Status","LoanAmount") \
.add_legend()
plt.show()  
sns.FacetGrid(df,hue="Property_Section",height=4) \
.map(plt.scatter,"ApplicantIncome","CoapplicantIncome") \
.add_legend()
plt.show()  
plt.figure(figsize = (10,7)) 
x = df["LoanAmount"] 
plt.hist(x, bins = 30, color = "pink") 
plt.title("Loan taken by Customers") 
plt.xlabel("Loan Figures") 
plt.ylabel("Count") 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
X=df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Married_Section',
        'Gender_Section','Edu_Section','Employed_Section','Property_Section']].values
y=df[["Loan_Section"]].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model.fit(X_train,y_train)
model.score(X_train,y_train)
model.score(X_test,y_test)
from sklearn import metrics
expected = y_test
predicted = model.predict(X_test)
print(metrics.classification_report(expected, predicted))
print(metrics.classification_report(expected, predicted))
