import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
data=pd.read_csv(' ')
data.head()
x=data.drop(columns=['Species'])
y=data.Species
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
RFC = RandomForestClassifier(n_estimators=100)
RFC = RFC.fit(x_train,y_train)
y_pred = RFC.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
