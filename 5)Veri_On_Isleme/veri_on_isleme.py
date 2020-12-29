#Aykırı Gözlem Analizi

#Aykırı Değerleri Yakalamak

import seaborn as sns
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()

df_table = df["table"]
df_table.head()

import matplotlib.pyplot as plt
sns.boxplot(x= df_table);
plt.show()

Q1 = df_table.quantile(0.25)
Q3 = df_table.quantile(0.75)
IQR = Q3 - Q1
Q1
Q3
IQR
alt_sinir = Q1 - 1.5*IQR
ust_sinir = Q3 + 1.5*IQR
alt_sinir
ust_sinir

(df_table < alt_sinir ) | (df_table > ust_sinir )

aykiri_tf = (df_table < alt_sinir )
aykiri_tf

df_table[aykiri_tf]

df_table[aykiri_tf].index

#Aykırı Değer Problemini Çözmek

df_table[aykiri_tf]

#Silme
import pandas as pd
type(df_table)
df_table =pd.DataFrame(df_table)

df_table.shape

t_df = df_table[~ ((df_table < (alt_sinir)) | (df_table > (ust_sinir))).any(axis = 1)] # ~ koşulu sağlamayanları alır.
t_df.shape

#Ortalama ile Doldurma
import seaborn as sns
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()

df_table = df["table"]
aykiri_tf.head()

df_table[aykiri_tf]
df_table.mean()

df_table[aykiri_tf] = df_table.mean()

df_table[aykiri_tf]

#Baskılama Yöntemi

import seaborn as sns
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df_table = df["table"]

df_table[aykiri_tf]

alt_sinir

df_table[aykiri_tf] = alt_sinir

df_table[aykiri_tf]

#Çok Değişkenli Aykırı Gözlem Analizi

import seaborn as sns
diamonds = sns.load_dataset('diamonds')
diamonds = diamonds.select_dtypes(include= ['float64', 'int64'])
df = diamonds.copy()
df = df.dropna()
df.head()

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors= 20, contamination= 0.1)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:10]

np.sort(df_scores[0:10])

np.sort(df_scores)[0:20]
np.sort(df_scores)[13]

esik_deger = np.sort(df_scores)[13]
aykiri_tf = df_scores > esik_deger

aykiri_tf

# Silme
yeni_df = df[df_scores > esik_deger] #Aykırı olmayanlar
yeni_df

df[df_scores < esik_deger] #aykırılar.

# Baskılama

df[df_scores == esik_deger]

baski_degeri = df[df_scores == esik_deger]

aykirilar = df[~aykiri_tf]

aykirilar
aykirilar.to_records(index = False)

res = aykirilar.to_records(index = False)
res[:] = baski_degeri.to_records(index = False)
res

df[~aykiri_tf]

import pandas as pd
df[~aykiri_tf] = pd.DataFrame(res, index= df[~aykiri_tf].index)
df[~aykiri_tf]

#Eksik Veri Analizi

#Hızlı Çözüm

import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
df = pd.DataFrame({"V1" : V1,
                   "V2" : V2,
                   "V3" : V3})
df

df.isnull().sum()
df.notnull().sum()

df.isnull().sum().sum()

df.isnull()

df[df.isnull().any(axis = 1)]

df[df.notnull().all(axis = 1)]

df[df["V1"].notnull() & df["V2"].notnull() & df["V3"].notnull()]

#Eksik Değerlerin Direkt Silinmesi

df.dropna()
#kalıcı için inplace

#basit değer atama

df["V1"]

df["V1"].mean()

#ortalama ile doldurma
df["V1"].fillna(df["V1"].mean())

df.apply(lambda x: x.fillna(x.mean()), axis= 0)

#Değer ile doldurma
df["V2"].fillna(0)

#Eksik Değerlerin Saptanması

#değişkenlerdeki tam değer sayısı
df.notnull().sum()

#değişkenlerdeki eksik değer sayısı
df.isnull().sum()

#veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()

#en az bir eksik değere sahip gözlemler
df[df.isnull().any(axis = 1)]

#tüm değerleri tam olan gözlemler
df[df.notnull().all(axis=1)]

#Eksik Veri Yapısının Görselleştirilmesi

import missingno as msno
import matplotlib.pyplot as plt
msno.bar(df);
plt.show()

msno.matrix(df);
plt.show()

df

import seaborn as sns
df = sns.load_dataset('planets')
df.head()

df.isnull().sum()

msno.matrix(df)
plt.show()

msno.heatmap(df);
plt.show()

#Silme Yöntemleri

import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
df = pd.DataFrame({"V1" : V1,
                   "V2" : V2,
                   "V3" : V3})
df

df.dropna()

df.dropna(how = "all") #hepsi aynı anda eksik ise siler

df.dropna(axis = 1) #Eksik gözlem olan değişkeni siler.

df.dropna(axis = 1, how = "all") #Tüm değerleri eksik olan değişkeni siler

#Basit Değer Atama
import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
df = pd.DataFrame({"V1" : V1,
                   "V2" : V2,
                   "V3" : V3})
df

#sayısal değer atama

df["V1"].fillna(0)
df

df["V1"].fillna(df["V1"].mean())

#Her değişkeni kendi ortalaması

#birinci yol
df.apply(lambda x: x.fillna(x.mean()), axis=0)

#ikinci yol
df.fillna(df.mean()[:])

#Ortalama medyan karışık atama için
df.fillna(df.mean()["V1":"V2"])
df["V3"].fillna(df["V3"].median())

#ucuncu yol

df.where(pd.notna(df), df.mean(), axis="columns")

#Kategorik Değişken Kırılımında Değer Atama
import numpy as np
import pandas as pd

V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4 = np.array(["IT","IT","IK","IK","IK","IK","IK","IT","IT"])

df = pd.DataFrame({"maas": V1, "V2": V2, "V3": V3, "departman": V4})
df

df.groupby("departman")["maas"].mean()

df["maas"]

df["maas"].fillna(df.groupby("departman")["maas"].transform("mean")) #Departman özelinde maaş ortalamasına göre doldurma

#Kategorik Değişkenler için Eksik Değer Atama

import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V4 = np.array(["IT",np.nan,"IK","IK","IK","IK","IK","IT","IT"], dtype=object)

df = pd.DataFrame(
    {"maas": V1,
     "departman": V4}
)

df

df["departman"].fillna(df["departman"].mode()[0])

df["departman"].fillna(method= "bfill") #kendinden sonra gelen değer ile doldurur.

df["departman"].fillna(method= "ffill") #bir önceki

#Tahmine Dayalı Değer Atama Yöntemleri

import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
df = sns.load_dataset('titanic')
df = df.select_dtypes(include=['float64', 'int64'])
print(df.head())
df.isnull().sum()

from ycimpute.imputer import knnimput
var_names = list(df)
n_df = np.array(df)
n_df[0:10]

n_df.shape

dff = knnimput.KNN(k = 4).complete(n_df)

type(dff)

dff = pd.DataFrame(dff, columns = var_names)
type(dff)

dff.isnull().sum()

#random forests

df = sns.load_dataset('titanic')
df = df.select_dtypes(include=['float64', 'int64'])
df.isnull().sum()

var_names = list(df)
n_df = np.array(df)

from ycimpute.imputer import iterforest
dff = iterforest.IterImput().complete(n_df) #no attribute hatası
dff = pd.DataFrame(dff, columns= var_names)
dff.isnull().sum()

#EM

df = sns.load_dataset('titanic')
df = df.select_dtypes(include=['float64', 'int64'])

from ycimpute.imputer import EM
var_names = list(df)
import numpy as np
n_df = np.array(df)

dff = EM().complete(n_df)
dff = pd.DataFrame(dff, columns= var_names)
dff.isnull().sum()

#Değişken Standardizasyonu (Veri Standardizasyonu)

import numpy as np
import pandas as pd
V1 = np.array([1,3,6,5,7])
V2 = np.array([7,7,5,8,12])
V3 = np.array([6,12,5,6,14])
df = pd.DataFrame(
    {"V1": V1,
     "V2": V2,
     "V3": V3})

df = df.astype(float)
df

#Standardizasyon

from sklearn import preprocessing
preprocessing.scale(df)
df

df.mean()

#Normalizasyon

preprocessing.normalize(df)

#Min-Max Dönüşümü

scaler = preprocessing.MinMaxScaler(feature_range= (10,20))
scaler.fit_transform(df)

#Değişken Dönüşümleri

import seaborn as sns
df = sns.load_dataset('tips')
df.head()

#0-1 Dönüşümü

from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()

lbe.fit_transform(df["sex"])

df["yeni_sex"] = lbe.fit_transform(df["sex"])
df

# 1 ve Diğerleri(0) Dönüşümü

df.head()

import numpy as np
df["yeni_day"] = np.where(df["day"].str.contains("Sun",1,0))
df

#Çok Sınıflı Dönüşüm
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()

lbe.fit_transform(df["day"])

#One - Hot Dönüşümü ve Dummy Değişken Tuzağı
import pandas as pd
df.head()
df_one_hot = pd.get_dummies(df, columns= ["sex"], prefix=["sex"])
df_one_hot.head()

df_one_hot = pd.get_dummies(df, columns= ["day"], prefix=["day"])
df_one_hot.head()

#Veri Standardizasyonu & Değişken DÖnüşümü

#Standartlaştırma

import numpy as np
import pandas as pd
V1 = np.array([1,3,6,5,7])
V2 = np.array([7,7,5,8,12])
V3 = np.array([6,12,5,6,14])

df = pd.DataFrame(
    {"V1" : V1,
     "V2" : V2,
     "V3" : V3}
)

df = df.astype(float)
df

from sklearn import preprocessing
preprocessing.scale(df)

#Normalizasyon
#0-1 aralığında dönüştürür.
preprocessing.normalize(df)

#Min-Max Dönüşümü
#İstenilen 2 aralık arasında dönüştürülür.
scaler = preprocessing.MinMaxScaler(feature_range= (10,20))
scaler.fit_transform(df)

#Binarize Dönüşüm
#Belirli eşik değerine göre 0-1 e dönüştürür.

binarizer = preprocessing.Binarizer(threshold=5).fit(df)
binarizer.transform(df)

#0-1 Dönüşümü
#Kategorik değişkeni sürekli değişkene çevirmek.

import seaborn as sns
tips = sns.load_dataset('tips')
df = tips.copy()
df_l = df.copy()

df_l.head()

df_l["yeni_sex"] = df_l["sex"].cat.codes
df_l.head()

lbe = preprocessing.LabelEncoder()
df_l["daha_yeni_sex"] = lbe.fit_transform(df_l["sex"])
df_l.head()

#1 ve Diğerleri(0) Dönüşümü
#İstanbul ve Diğerleri gibi sınıflandırma dönüşümleri

df.head()

df_l.head()
df_l["yeni_gun"] = np.where(df_l["day"].str.contains("Sun"),1,0)
df_l.head(20)

#Çok Sınıflı Dönüşüm
lbe = preprocessing.LabelEncoder()
df_l["daha_yeni_gun"] = lbe.fit_transform(df_l["day"])
df_l

#Sınıf sayısı kadar böler.

#One-Hot Dönüşümü ve Dummy Değişken Tuzağı

df_one_hot = df.copy()
pd.get_dummies(df_one_hot, columns= ["sex"], prefix= ["sex"]).head()

pd.get_dummies(df_one_hot, columns= ["day"], prefix= ["day"]).head()

#Sürekli Değişkeni Kategorik Değişkene Çevirme
df.head()
dff = df.select_dtypes(include= ["float64", "int64"])
est = preprocessing.KBinsDiscretizer(n_bins= [3,2,2], encode= "ordinal", strategy="quantile").fit(dff)
est.transform(dff)[0:10]

#Değişkeni İndexe, İndexi Değişkene ÇEvirmek
df.head()

df["yeni_degisken"] = df.index
df

df["yeni_degisken"] = df["yeni_degisken"] + 10
df.head()
df.index = df["yeni_degisken"]
df.index

