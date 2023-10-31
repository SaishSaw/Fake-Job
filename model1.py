import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,accuracy_score,f1_score,recall_score,confusion_matrix
import os
print("Current working directory:", os.getcwd())

#Reading the dataset.
fake = pd.read_csv('fake_job_postings.csv',na_values = '?')
#Filling missing values with blank values.
fake.fillna('',inplace=True)

#Importing preprocessed tokens directly from pickle file.
with open('lsa_summary.pkl','rb') as file:
    summary = pickle.load(file)

#Creating necessary changes in dataset for sending to model building.
fake.drop(['requirements','company_profile','description','benefits','title'],axis=1,inplace=True)

#Extracting numeric output from pickle file.
numeric_summary = []
for List in summary:
    for innerList in List:
        number = innerList[1]
        numeric_summary.append(number)

#Creating modified dataframe.
#Concatenating the numeric values for text data in the dataframe.
num_df = pd.DataFrame(numeric_summary,columns=['Vec_output'])
print('=========================================================')
print('The dataframe after concatenating.')
final_df = pd.concat([fake,num_df],axis=1)     

##Majority of attributes have high % of NA values. Hence no point in imputation.
final_df.drop(['department','salary_range','telecommuting','required_experience','required_education','function'],axis=1,inplace=True)

#extracting country from location.
def extract_country(text):
    if isinstance(text,str):
        txt = text.split(",")
        extract = txt[0]
        return extract
    
#Imputing the dataframes incase of any missing values.
impute = SimpleImputer(strategy='most_frequent')
imputed_data = impute.fit_transform(final_df)
imputed_df = pd.DataFrame(imputed_data,columns=final_df.columns)

# Converting object to numeric.
le = LabelEncoder()
imputed_df['industry'] = le.fit_transform(imputed_df['industry']).ravel()
imputed_df['employment_type'] = le.fit_transform(imputed_df['employment_type']).ravel()
imputed_df['location'] = le.fit_transform(imputed_df['location']).ravel()

for i in imputed_df.columns[1:8]:
    if imputed_df[i].dtype == 'object':
        imputed_df[i] = pd.to_numeric(imputed_df[i],errors='coerce')

#spliting the dataset into train test split for building model.
X = imputed_df.drop(['fraudulent','job_id'],axis=1)
Y = imputed_df['fraudulent']
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=123,shuffle=True)
xtrain.fillna(0.103346,inplace=True)
ytrain = le.fit_transform(ytrain)

#Using synthetic sample generation method to tackle the class imbalance.
smt = SMOTE(random_state=42)
xtrain_sm,ytrain_sm = smt.fit_resample(xtrain,ytrain)

#Model building.
rdf = RandomForestClassifier(random_state=10,criterion='gini',max_depth=12,min_samples_split=3\
                            ,max_features='log2',n_estimators=150)
rdf.fit(xtrain_sm,ytrain_sm)

#selecting best threshold obtained in training.
yprob_rf = rdf.predict_proba(xtest)[:,1]
threshold = 0.2
y_pred = [1 if prob >= 0.2 else 0 for prob in yprob_rf]
#obtaining recall scores.
recall = recall_score(ytest,y_pred)
print(recall)

import pickle
with open('model.pkl','wb') as file:
    pickle.dump(rdf,file)


    
