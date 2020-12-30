#Basit Doğrusal Regresyon

import pandas as pd
ad = pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//6)Makine_ogrenmesi/1)Dogrusal_Regresyon/Advertising.csv")
df = ad.copy()
df.head()

df= df.iloc[:,1:len(df)] #indexi değişken olarak okuduğu için yaptık. (veri setinde usecols = [1,2,3,4] ile de yapılabilir
df.head()

df.info()

df.describe().T

df.isnull().values.any()

df.corr()

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, kind="reg");
plt.show()

sns.jointplot(x = "TV", y= "sales", data=df, kind="reg")
plt.show()

#Statsmodles ile modelleme

import statsmodels.api as sm
X = df[["TV"]]
X[0:5]

X = sm.add_constant(X)
X[0:5]

y = df["sales"]
y[0:5]

lm = sm.OLS(y, X)
model = lm.fit()
model.summary()

import statsmodels.formula.api as smf
lm = smf.ols("sales ~ TV",df)
model = lm.fit()
model.summary()

model.params

model.summary().tables[1]

model.conf_int()

model.f_pvalue

print("f_pvalue: ","%.4f" % model.f_pvalue)

print("fvalue: ","%.4f" % model.fvalue)

print("tvalue: ","%.2f" % model.tvalues[0:1])

model.mse_model
#Sonuç baya kötü

model.rsquared

model.rsquared_adj

model.fittedvalues[0:5]

y[0:5]

#Modelin denklemi
print("Sales = " + str("%.2f" %model.params[0]) +" +TV" + "*" + str("%.2f" % model.params[1]))

g = sns.regplot(df["TV"], df["sales"], ci = None, scatter_kws={'color': 'r','s':9})
g.set_title("Model Denklemi: Sales = 7.03 + TV*0.05")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom = 0);
plt.show()

from sklearn.linear_model import LinearRegression
X = df[["TV"]]
y = df["sales"]
reg = LinearRegression()
model = reg.fit(X,y)

model.intercept_
model.coef_

model.score(X,y)

model.predict(X)[0:10]

#Tahmin

#30 birim Tv harcaması olduğunda satışların tahmini ne olur?
7.03 + 30*0.04

X = df[["TV"]]
y = df["sales"]
reg = LinearRegression()
model = reg.fit(X,y)
model.predict([[30]])

yeni_veri = [[5],[90],[200]]
model.predict(yeni_veri)

