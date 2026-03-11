import numpy as np
from sklearn.linear_model import LogisticRegression
x=[[2],[4],[5],[6]]
y=[0,0,1,1]
m=LogisticRegression().fit(x,y)
print("prediction",m.predict([[3]]))
print("probability",m.predict_proba([[3]]))

