from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans

#K-Means

df = pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//6)Makine_ogrenmesi/4)gozetimsiz_ogrenme/USArrests.csv")
df.head()

df.index = df.iloc[:,0]
df.head()

df = df.iloc[:,1:5]

df.head()

df.isnull().sum()

df.describe().T

df.hist(figsize= (10,10));
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 4)

kmeans.fit(df)
kmeans.n_clusters

kmeans.cluster_centers_

kmeans.labels_

#görselleştirme
kmeans = KMeans(n_clusters=2)
k_fit = kmeans
k_fit.fit(df)

kumeler = k_fit.labels_
plt.scatter(df.iloc[:,0], df.iloc[:,1], c= kumeler, s= 50, cmap="viridis")
merkezler = k_fit.cluster_centers_
plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha=0.5);
plt.show()

from mpl_toolkits.mplot3d import Axes3D

kmeans = KMeans(n_clusters=3)
k_fit = kmeans
k_fit.fit(df)
kumeler = k_fit.labels_
merkezler = kmeans.cluster_centers_

plt.rcParams['figure.figsize'] = (16,9)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:,2]);
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c = kumeler)
ax.scatter(merkezler[:,0], merkezler[:,1], merkezler[:,2],
           marker="*", c= "#050505", s=1000);
plt.show()

#kümeler ve gözlem birimleri

kmeans = KMeans(n_clusters=3)
k_fit = kmeans
k_fit.fit(df)
kumeler = k_fit.labels_

pd.DataFrame({"Eyaletler": df.index, "Kumeler": kumeler})[0:10]

df["kume_no"] = kumeler
df.head()

#Optimum Kume Sayısının Belirlenmesi

from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,20))
visualizer.fit(df)
visualizer.poof()

kmeans = KMeans(n_clusters= 4)
k_fit = kmeans.fit(df)
kumeler = k_fit.labels_

pd.DataFrame({"Eyaletler": df.index, "Kumeler": kumeler})[0:10]

#Hiyerarşik Kümeleme

from scipy.cluster.hierarchy import linkage
hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")
hc_single = linkage(df, "single")

from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15,10))
plt.title('Hiyerarşik Kümeleme - Dendogram')
plt.xlabel('Indexler')
plt.ylabel('Uzaklık')
dendrogram(hc_complete, leaf_font_size=10);
plt.show()

from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15,10))
plt.title('Hiyerarşik Kümeleme - Dendogram')
plt.xlabel('Indexler')
plt.ylabel('Uzaklık')
dendrogram(hc_complete, truncate_mode="lastp", p =12, show_contracted= True);
plt.show()

#Optimum Küme Sayısı

from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15,10))
plt.title('Hiyerarşik Kümeleme - Dendogram')
plt.xlabel('Indexler')
plt.ylabel('Uzaklık')
den = dendrogram(hc_complete, leaf_font_size=10);
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters= 4, affinity="euclidean", linkage="ward")
cluster.fit_predict(df)

pd.DataFrame({"Eyaletler": df.index, "Kumeler": cluster.fit_predict(df)})[0:10]
df["kume_no"] = cluster.fit_predict(df)
df.head()

#PCA

df.head()

from sklearn.preprocessing import StandardScaler
df = StandardScaler().fit_transform(df)
df[0:5,0:5]

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

bilesen_df = pd.DataFrame(data= pca_fit, columns=["birinci_bilesen", "ikinci_bilesen","ucuncu_bilesen"])
bilesen_df.head()

pca.explained_variance_ratio_

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

