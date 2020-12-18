#Neden Numpy?

a = [1,2,3,4]
b = [2,3,4,5]

import numpy as np
a = np.array([1,2,3,4])
b = np.array([2,3,4,5])

a*b

#NumPy Array'i Oluşturmak

a = np.array([1,2,3,4])
type(a)
np.array([3.14,4,2,13], dtype="float32")
np.array([3.14,4,2,13], dtype="int")

#sifirdan array Oluşturma

np.zeros(10, dtype=int)

np.ones((3,5),dtype=int)

np.full((3,5),3)

np.arange(0,31,3)

np.linspace(0,1,10)

np.random.normal(10,4,(3,4))

np.random.randint(0,10,(3,3))

#Özellikleri

a = np.random.randint(10,size=10)

a.ndim

a.shape

a.size

a.dtype

b= np.random.randint(10,size=(3,5))

b

b.ndim

b.shape
b.size
b.dtype

#Reshaping

np.arange(10)

np.arange(1,10).reshape((3,3))

a = np.arange(1,10)
a
a.ndim

b = a.reshape((1,9))
b.ndim

#Array Birleştirme (Concatenation)

x = np.array([1,2,3])
y = np.array([4,5,6])

np.concatenate([x,y])

z = np.array([7,8,9])
np.concatenate([x,y,z])

#iki boyut

a = np.array([[1,2,3],
              [4,5,6]])
np.concatenate([a,a])

np.concatenate([a,a], axis=1)

#Array Ayırma (Splitting)

x = np.array([1,2,3,99,99,3,2,1])

np.split(x, [3,5])

a,b,c = np.split(x, [3,5])

a
b
c

#İki Boyutlu Ayırma

m = np.arange(16).reshape(4,4)
m

np.vsplit(m, [2])
ust, alt = np.vsplit(m, [2])
ust
alt

m
np.hsplit(m, [2])

sag, sol = np.hsplit(m, [2])

sag
sol

#Sıralama (Sorting)

v = np.array([2,1,4,3,5])
v

np.sort(v)
v
v.sort()
v

#iki boyutlu array sıralama

m = np.random.normal(20,5, (3,3))
m

np.sort(m, axis=1)

np.sort(m, axis=0)


#Index ile Elemanlara erişmek

a = np.random.randint(10, size=10)
a
a[0]

m = np.random.randint(10, size=(3,5))
m

m[0,0]
m[1,2]
m[1,4] = 2.2
m

#Array Alt Küme Elemanlara Erişmek (Array Alt Kümesine Erişmek)

a = np.arange(20,30)
a

a[0:3]

a[:3]

a[3:]

a[1::2]

a[0::2]

a[2::2]

a[0::3]

#İki Boyutlu Slice İşlemleri
m = np.random.randint(10, size=(5,5))
m

m[:,0]

m[:,1]

m[:,4]

m
m[0,:]
m[0]

m[1,:]

m[0:2, 0:3]

m[::,:2]

m[:,0:2]
m
m[1:3, 0:2]

#slicing ile Elemanlara Erişmek(Array Alt Kümesine Erişmek)

import numpy as np
a = np.arange(20,30)
a

a[0:3]

a[:3]

a[3:]

a[1::2]

a[0::2]


a[2::2]

#iki boyutlu slice işlemleri

m = np.random.randint(10, size = (5,5))
m
m[:,0]

m[:,1]

m[:,4]

m
m[0,:]

m[0]

m[1,:]

m[0:2, 0:3]

m[:,0:2]

m[1:3, 0:2]

m

#Alt Küme Üzerinde İşlem Yapmak

a = np.random.randint(10, size= (5,5))
a

alt_a = a[0:3,0:2]
alt_a

alt_a[0,0] = 9999
alt_a[1,1] = 888

alt_a

a

#Alt küme değişikliği ana parçayı da etkiler

m = np.random.randint(10, size= (5,5))
m

alt_m = m[0:3,0:2].copy()
alt_m

alt_m[0,0] = 999
alt_m
m

#copy ile yapılırsa ana parçayı etkilemez.

#Fancy Index ile Elemeanlara Erişmek

v = np.arange(0,30,3)
v

v[1]

v[3]

v[5]

[v[1],v[3],v[5]]

al_getir = [1,3,5]
v
v[al_getir] #fancy index budur.

#iki boyutta fancy

m = np.arange(9).reshape((3,3))
m

satir = np.array([0,1])
sutun = np.array([1,2])

m[satir,sutun]

#basit index ile facny index

m

m[0,[1,2]]

#slice ile fancy

m[0:, [1,2]]

#Koşullu Eleman İşlemleri

v = np.array([1,2,3,4,5])

v <3

v[v < 3]

v[v > 3]

v[v >=3 ]
v[v <= 3]
v[v == 3]
v[v != 3]

v

v*2

v/5

v*5/10

v**2

#Matematiksel İşlemler

v = np.array([1,2,3,4,5])
v

v-1

v*5

v/5

v*5/10 -1

#ufunc

np.subtract(v,1)
np.add(v,1)
np.multiply(v,4)
np.divide(v,3)

v**2
v**3
np.power(v,3)

v%2
np.mod(v,2)

np.absolute(np.array(-3))

np.sin(360)

np.cos(180)

v = np.array([1,2,3])
np.log(v)

np.log2(v)

np.log10(v)

#İstatiksel Hesaplamalar

np.mean(v,axis=0) #Returns mean along specific axis
v.sum() # Returns sum of arr
v.min() # Returns minimum value of arr
v.max(axis=0) # Returns maximum value of specific axis
np.var(v) # Returns the variance of array
np.std(v,axis=1) # Returns the standard deviation of specific axis
v.corrcoef() # Returns correlation coefficient of array


#Numpy ile İki Bilinmeyenli Denklem Çözümü

import numpy as np
5*x0+x1 = 12
x0+3*x1 = 10
"""
"""
a = np.array([[5,1], [1,3]])
b = np.array([12,10])
a
b

x = np.linalg.solve(a, b)
x


#PANDAS

#Pandas Serisi Oluşturmak

import pandas as pd

pd.Series([1,2,3,4,5])

seri = pd.Series([1,2,3,4,5])

type(seri)

seri.axes
seri.dtype
seri.size
seri.ndim
seri.values
seri.head(3)
seri.tail(3)

#index isimlendirmesi

pd.Series([99,12,33,445,55])

pd.Series([99,12,33,445,55],index=[1,3,4,5,6])
pd.Series([99,12,33,445,55],index=["a","b","c","d","e"])
seri["a"]
seri["a":"c"]

#sozlük üzerinden liste oluşturmak

sozluk = pd.Series({"reg":10, "log":11, "cart":12})
sozluk

#iki seriyi birlestirerek seri oluşturma

pd.concat([seri,seri])

#Eleman İşlemleri
import numpy as np
a = np.array([11,22,33,455,56])
seri = pd.Series(a)
seri

seri[0]
seri[0:3]

seri = pd.Series([121,200,150,99], index= ["reg","loj","cart","rf"])

seri
seri.index

seri.keys()

list(seri.items())
seri.values

#eleman sorgulama

"reg" in seri

"a" in seri

seri["reg"]

#fancy eleman

seri[["rf","reg"]]

seri["reg"] = 130
seri["reg"]

seri["reg":"loj"]

#Pandas DataFrame Oluşturma

import pandas as pd
l = [1,2,39,67,90]
l
pd.DataFrame(l, columns=["degisken_ismi"])

import numpy as np
m = np.arange(1,10).reshape(3,3)
m
pd.DataFrame(m, columns=["var1","var2","var3"])

#df isimlendirme
df = pd.DataFrame(m, columns=["var1","var2","var3"])
df.head

df.columns = ("deg1","deg2","deg3")

type(df)

df.axes
df.shape
df.ndim
df.size
df.values
type(df.values)
df.head()
df.tail()

a = np.array([1,2,3,4,5])
pd.DataFrame(a, columns= ["deg1"])

#Eleman İşlemleri

s1 = np.random.randint(10, size=5)
s2 = np.random.randint(10, size=5)
s3 = np.random.randint(10, size=5)

sozluk = {"var1":s1, "var2": s2, "var3": s3}
sozluk

df = pd.DataFrame(sozluk)
df

df[0:1]

df.index

df.index = ["a","b","c","d","e"]
df

df["c":"d"]

#silme

df.drop("a",axis=0)
df

df.drop("a",axis=0, inplace= True)
df

#fancy
l = ["c", "e"]
df.drop(l, axis=0)

#degiskenler icin

"var1" in df

l = ["var1","var2","var4"]
for i in l:
    print(i in df)

df

df["var4"] = df["var1"] / df["var2"]
df

#degisken silmek

df.drop("var4",axis=1)
df

df.drop("var4",axis=1,inplace=True)
df

l = ["var1","var2"]
df.drop(l,axis=1)
df

#Gözlem ve Değişken Seçimi
m = np.random.randint(1,30, size=(10,3))
df = pd.DataFrame(m, columns= ["var1","var2","var3"])
df

#loc: tanımlandığı şekliyle seçim yapmak için kullanılır

df.loc[0:3]

#iloc: alışık olduğumuz indexleme mantığıyla seçim yapar.

df.iloc[0:3]

df.iloc[0,0]
df.iloc[:3,:2]

df.loc[0:3,"var3"]
                    #Satırlarla ilgili mutlak değer işaretlemesinde loc kullanılır. İloc burada aşağıdaki hatayı verir.
df.iloc[0:3,"var3"]

df.iloc[0:3]["var3"] #Hatadan kaçmak için bu olabilir ama loc daha mantıklıdır.

#Koşullu Eleman İşlemleri
df[df.var1 > 15] ["var1"]

df[(df.var1 > 15) & (df.var3 < 5)]

df.loc[(df.var1 > 15), ["var1","var2"]]

df[(df.var1 > 15)] [["var1","var2"]]

#Birleştirme(Join) İşlemleri

df2 = df+99
df2

pd.concat([df,df2])
#index problemini giderir.
pd.concat([df,df2], ignore_index= True)

df.columns
df2.columns

df2.columns = ["var1","var2","deg3"]
df2
df
pd.concat([df,df2])

pd.concat([df,df2], join= "inner") #kesişimlere göre birleştirdi.

#İleri Birleştirme İşlemleri

#birebir birleştirme

df1 = pd.DataFrame({'calisanlar':["ali","veli","ayşe","fatma"],
                    'grup':["muhasebe", "muhendislik", "muhendislik", "İK"]})

df1

df2 = pd.DataFrame({'calisanlar': ["ayşe","ali","veli","fatma"],
                    'ilk_giris': [2010,2009,2014,2019]})

df2

pd.merge(df1,df2)

pd.merge(df1,df2,on="calisanlar")

#Çoktan tekile birleştirme

df3 = pd.merge(df1,df2)
df3

df4 = pd.DataFrame({"grup": ["muhasebe", "muhendislik", "İK"],
                    "mudur": ["caner","mustafa","berkcan"]})
df4

pd.merge(df3,df4)

#çoktan çoka

df5 = pd.DataFrame({'grup': ["muhasebe", "muhasebe", "muhendislik", "muhendislik", "İK", "İK"],
                    'yetenekler': ["matematik", "excel", "kodlama", "linux", "excel", "yonetim"]})
df5

pd.merge(df1,df5)

#Toplulaştırma ve Gruplama

import seaborn as sns
df = sns.load_dataset("planets")
df.head()

df.shape

df.mean()

df["mass"].mean()

df["mass"].min()

df["mass"].max()
df["mass"].sum()
df["mass"].std()
df["mass"].var()

df.describe().T

df.dropna().describe().T

#Gruplama İşlemleri
import pandas as pd
df = pd.DataFrame({'gruplar': ["A","B","C","A","B","C"],
                   'veri': [10,11,52,23,43,55]},columns=['gruplar', 'veri'])

df

df.groupby("gruplar")
df.groupby("gruplar").mean()
df.groupby("gruplar").sum()

import seaborn as sns
df = sns.load_dataset("planets")
df.head()

df.groupby("method")
df.groupby("method")["orbital_period"]
df.groupby("method")["orbital_period"].mean()

df.groupby("method")["mass"].mean()

#İleri Toplulaştırma(Aggregate, filter, transform, apply)
import numpy as np
df = pd.DataFrame({"gruplar": ["A","B","C","A","B","C"],
                   "degisken1": [10,23,33,22,11,99],
                   "degisken2": [100,253,333,252,111,969]},
                  columns= ["gruplar", "degisken1", "degisken2"])

df

#aggregate
df.groupby("gruplar").mean()

df.groupby("gruplar").aggregate(["min", np.median, max])

df.groupby("gruplar").aggregate({"degisken1": "min", "degisken2": "max"})

#Filter
df = pd.DataFrame({"gruplar": ["A","B","C","A","B","C"],
                   "degisken1": [10,23,33,22,11,99],
                   "degisken2": [100,253,333,252,111,969]},
                  columns= ["gruplar", "degisken1", "degisken2"])
df

def filter_func(x):
    return x["degisken1"].std() > 9

df.groupby("gruplar").std()
df.groupby("gruplar").filter(filter_func)

#Transform

df = pd.DataFrame({"gruplar": ["A","B","C","A","B","C"],
                   "degisken1": [10,23,33,22,11,99],
                   "degisken2": [100,253,333,252,111,969]},
                  columns= ["gruplar", "degisken1", "degisken2"])
df

df["degisken1"]*9

df_a = df.iloc[:,1:3]
df_a.transform(lambda x: (x-x.mean()) / x.std())

#Apply

df = pd.DataFrame({"degisken1": [10,23,33,22,11,99],
                   "degisken2": [100,253,333,252,111,969]},
                  columns= ["degisken1", "degisken2"])

df

df.apply(np.sum)
df.apply(np.mean)

#Pivot Tablolar

import pandas as pd
import numpy as  np
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.head()

titanic.groupby("sex")
titanic.groupby("sex")["survived"].mean()
titanic.groupby("sex")[["survived"]].mean()
titanic.groupby(["sex","class"])[["survived"]].aggregate("mean").unstack()

#pivot ile table

titanic.pivot_table("survived", index="sex", columns="class")

titanic.age.head()
age = pd.cut(titanic["age"], [0, 18, 90])
age.head(10)

titanic.pivot_table("survived", ["sex",age], "class")

#Dış Kaynaklı Veri Okumak
import pandas as pd
pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//2)Veri_Manipulasyonu_Numpy&Pandas//ornekcsv.csv", sep = ";")

pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//2)Veri_Manipulasyonu_Numpy&Pandas//duz_metin.txt")

pd.read_excel("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//2)Veri_Manipulasyonu_Numpy&Pandas//ornekx.xlsx")

