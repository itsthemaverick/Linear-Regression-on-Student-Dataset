import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 

data =  pd.read_csv("data/large_study_data.csv")

X = data[['Hours_Studied']]
y = data['Marks']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2,test_size=0.2)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred= model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)

print("Slope (m) : ",model.coef_[0])
print("Intercept (c) : ",model.intercept_)
print("Error : ",mse)


print("Predicted marks for 11 hours:",
      model.predict([[11]])[0])