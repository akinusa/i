from sklearn.tree import DecisionTreeClassifier,plot_tree
import numpy as np
import matplotlib.pyplot as plt
x=np.array([[20,20000],[25,25000],[35,35000],[40,40000]])

y=np.array([0,0,1,1])
m=DecisionTreeClassifier().fit(x,y)
p=m.predict([[30,30000]])
print(p)
plot_tree(m,filled=True)
plt.show()
