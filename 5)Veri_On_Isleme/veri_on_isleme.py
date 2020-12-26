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

