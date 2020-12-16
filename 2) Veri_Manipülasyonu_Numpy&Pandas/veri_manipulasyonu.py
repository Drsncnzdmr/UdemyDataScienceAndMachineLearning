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

"""
5*x0+x1 = 12
x0+3*x1 = 10
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

