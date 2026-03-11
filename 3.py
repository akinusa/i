import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data={'Income':[15,16,17,18,50,52,54,55,90,92],
      'Score':[39,81,6,77,40,42,60,61,20,19]}

df=pd.DataFrame(data)

k=KMeans(n_clusters=3)
df['Cluster']=k.fit_predict(df)

plt.scatter(df['Income'],df['Score'],c=df['Cluster'])
plt.scatter(k.cluster_centers_[:,0],k.cluster_centers_[:,1],marker='X',s=200)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Customer Clusters")
plt.show()
