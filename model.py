import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
import pickle

df = pd.read_csv("diabetes.csv")
X = df.iloc[:,1:5]
y = df.iloc[:,8:]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

ridge = RidgeClassifier()

ridge.fit(X_train,y_train)

pickle.dump(ridge,open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))

print(model.predict([[148,72,35,0]]))
