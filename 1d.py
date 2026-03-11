import numpy as np
from sklearn.neighbors import KNeighborsClassifier
x=[[3,5],[4,4],[6,4],[7,3]]
y=[0,0,1,1]
m=KNeighborsClassifier(n_neighbors=3).fit(x,y)
print(m.predict([[5,4]]))
      
