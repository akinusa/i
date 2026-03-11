import numpy as np
from sklearn.linear_model import LogisticRegression

x=np.array([[2],[4],[6],[8]])
y=np.array([0,0,1,1])

m=LogisticRegression().fit(x,y)
print(m.predict([[5]]))
