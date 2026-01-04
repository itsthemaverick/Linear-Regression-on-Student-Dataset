import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/large_study_data.csv")

sample = data.sample(500)

X = sample[['Hours_Studied']]
y = sample['Marks']

model = LinearRegression()

model.fit(data[['Hours_Studied']],data['Marks'])

plt.scatter(X,y)
plt.plot(X,model.predict(X))
plt.xlabel("hours studied")
plt.ylabel("marks")
plt.title("Linear Regression")
plt.show()
plt.savefig('LR.png')