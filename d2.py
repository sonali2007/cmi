import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing

def preprocess_features(df):

	df = pd.DataFrame(df,columns=['Age','Income','Gender','Marital Status'])
	le = preprocessing.LabelEncoder()
	df=df.apply(le.fit_transform)
	print('\n')
	print(df)
	print('\n')
	return(df)
	
	
    
df = pd.read_csv('cosmetics.csv')
print('\n\n')
print(df.head())
df.drop(columns=['ID'],inplace= True)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
print("\n\nFeatures :")
print(x.head())
print("\n\nLabels :")
print(y.head())
x = preprocess_features(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state=0)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)
print("\n\nAccuracy :",model.score(x_test,y_test))
test = pd.DataFrame({'Age':'<21','Income':'Low','Gender':'Female','Marital Status':'married'},index=[0])
test = preprocess_features(test)
print("\n\nPrediction for Test Value ",model.predict(test))
tree.export_graphviz(model,out_file='tree.dot',rounded=True,feature_names=['Age','Income','Gender','Marital Status'],filled=True,class_names=['No','Yes'])


