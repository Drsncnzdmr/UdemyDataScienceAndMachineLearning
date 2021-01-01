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

#Artıklar ve Makine Öğrenmesindeki Önemi

from sklearn.metrics import mean_squared_error, r2_score
lm = smf.ols("sales ~ TV",df)
model = lm.fit()
mse = mean_squared_error(y,model.fittedvalues)
mse

import numpy as np
rmse = np.sqrt(mse)
rmse

reg.predict(X)[0:10]

k_t = pd.DataFrame({"gercek_y": y[0:10],
                    "tahmin_y": reg.predict(X)[0:10]})
k_t

k_t["hata"] = k_t["gercek_y"] - k_t["tahmin_y"]
k_t

k_t["hata_kare"] = k_t["hata"] **2
k_t

np.sum(k_t["hata_kare"])

np.mean(k_t["hata_kare"])

np.sqrt(np.mean(k_t["hata_kare"]))

#Çoklu Doğrusal Regresyon

import pandas as pd
ad = pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//6)Makine_ogrenmesi/1)Dogrusal_Regresyon/Advertising.csv", usecols = [1,2,3,4])
df = ad.copy()
df.head()

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

X = df.drop("sales", axis = 1)
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.20, random_state= 42)

X_train.shape
y_train.shape

X_test.shape
y_test.shape

training = df.copy()
training.shape

# Statsmodels

import statsmodels.api as sm
lm = sm.OLS(y_train,X_train)

model = lm.fit()
model.summary()

model.summary().tables[1]

#Scikit-learn model

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train,y_train)
model.intercept_
model.coef_

#Tahmin

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T
model.predict(yeni_veri)

import numpy as np
rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
rmse

rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
rmse

#Model Tuning/Model Doğrulama
df.head()

X = df.drop('sales', axis=1)
y = df["sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size= 0.20,
                                                    random_state=144)
lm = LinearRegression()
model = lm.fit(X_train, y_train)

np.sqrt(mean_squared_error(y_train, model.predict(X_train)))

model.score(X_train, y_train)

cross_val_score(model, X, y, cv = 10, scoring= "r2").mean()

cross_val_score(model, X_train,y_train,cv=10,scoring="r2").mean()

-cross_val_score(model, X_train,y_train,cv=10,scoring="neg_mean_squared_error") #Sonuçların pozitif olması için - koyuldu

np.sqrt(-cross_val_score(model, X_train,y_train,cv=10,scoring="neg_mean_squared_error"))

np.sqrt(-cross_val_score(model, X_train,y_train,cv=10,scoring="neg_mean_squared_error")).mean()

np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

np.sqrt(-cross_val_score(model, X_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean()

#PCR Model
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

hit = pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//6)Makine_ogrenmesi/1)Dogrusal_Regresyon/Hitters.csv")
df = hit.copy()
df = df.dropna()
df.head()

df.info()

df.describe().T

dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
dms.head()

y = df["Salary"]
X_ = df.drop(["Salary", 'League', "Division", "NewLeague"], axis = 1).astype("float64")
X_.head()
X = pd.concat([X_, dms[['League_N', "Division_W", "NewLeague_N"]]], axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)

print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

training = df.copy()
print("training", training.shape)

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
pca = PCA()

X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_train[0:1,:]

import numpy as np
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)[0:5]

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

pcr_model = lm.fit(X_reduced_train,y_train)

pcr_model.intercept_
pcr_model.coef_

#PCR Tahmin
y_pred = pcr_model.predict(X_reduced_train)
y_pred[0:5]
np.sqrt(mean_squared_error(y_train,y_pred))

df["Salary"].mean()
r2_score(y_train,y_pred)

pca2 = PCA()

X_reduced_test = pca2.fit_transform(scale(X_test))
y_pred = pcr_model.predict(X_reduced_test)

np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning

lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train[:,0:10], y_train)
y_pred = pcr_model.predict(X_reduced_test[:,0:10])
print(np.sqrt(mean_squared_error(y_test,y_pred)))

from sklearn import model_selection

cv_10 = model_selection.KFold(n_splits= 10,
                              shuffle= True,
                              random_state=1)
lm = LinearRegression()
RMSE = []

for i in np.arange(1, X_reduced_train.shape[1]+1):
    score = np.sqrt(-1*model_selection.cross_val_score(lm,
                                                       X_reduced_train[:,:i],
                                                       y_train.ravel(),
                                                       cv = cv_10,
                                                       scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

import matplotlib.pyplot as plt
plt.plot(RMSE, '-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Maaş Tahmin Modeli İçin PCR Model Tuning');
plt.show()

lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train[:,0:6], y_train)
y_pred = pcr_model.predict(X_reduced_train[:,0:6])
print(np.sqrt(mean_squared_error(y_train,y_pred)))

y_pred = pcr_model.predict(X_reduced_test[:,0:6])
print(np.sqrt(mean_squared_error(y_test,y_pred)))


#PLS

#Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
hit = pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//6)Makine_ogrenmesi/1)Dogrusal_Regresyon/Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
from sklearn.cross_decomposition import PLSRegression, PLSSVD

pls_model = PLSRegression().fit(X_train,y_train)
pls_model.coef_

#Tahmin

pls_model


pls_model.predict(X_train)[0:10]

y_pred = pls_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))
r2_score(y_train,y_pred)

y_pred = pls_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning

#CV
cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

#Hata hesaplamak için döngü
RMSE = []

for i in np.arange(1, X_train.shape[1]+1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls,X_train,y_train,cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

#Sonuçların Görselleştirilmesi
plt.plot(np.arange(1, X_train.shape[1] +1), np.array(RMSE), '-v', c= "r")
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Salary')
plt.show()

pls_model = PLSRegression(n_components= 2).fit(X_train,y_train)
y_pred = pls_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Ridge Regresyon

#Model

hit = pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//6)Makine_ogrenmesi/1)Dogrusal_Regresyon/Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha= 0.1).fit(X_train, y_train)

ridge_model.coef_

lambdalar = 10**np.linspace(10,-2,100)*0.5

ridge_model = Ridge()
katsayilar = []

for i in lambdalar:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(X_train,y_train)
    katsayilar.append(ridge_model.coef_)

ax = plt.gca()
ax.plot(lambdalar, katsayilar)
ax.set_xscale('log')

plt.xlabel('Lambda(Alpha) Değerleri')
plt.ylabel('Katsayılar/Ağırlıklar')
plt.title('Düzenlileştirmenin Bir Fonksiyonu Olarak Ridge Katsayıları');
plt.show()

#Tahmin

y_pred = ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning

lambdalar = 10**np.linspace(10,-2,100)*0.5
lambdalar[0:5]

from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas= lambdalar, scoring="neg_mean_squared_error",normalize= True)
ridge_cv.fit(X_train,y_train)
ridge_cv.alpha_

ridge_tuned = Ridge(alpha=ridge_cv.alpha_, normalize= True).fit(X_train,y_train)
np.sqrt(mean_squared_error(y_test,ridge_tuned.predict(X_test)))

#Lasso Regresyon

#Model

hit = pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//6)Makine_ogrenmesi/1)Dogrusal_Regresyon/Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.1).fit(X_train,y_train)
lasso_model
lasso_model.coef_

lasso = Lasso()
lambdalar = 10**np.linspace(10,-2,100)*0.5
katsayilar = []

for i in lambdalar:
    lasso.set_params(alpha = i)
    lasso.fit(X_train,y_train)
    katsayilar.append(lasso.coef_)

ax = plt.gca()
ax.plot(lambdalar*2, katsayilar)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()

#Tahmin

lasso_model.predict(X_test)
y_pred = lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#Tuning

from sklearn.linear_model import LassoCV
lasso_cv_model = LassoCV(alphas= None, cv=10, max_iter=10000, normalize= True)
lasso_cv_model.fit(X_train, y_train)
lasso_cv_model.alpha_

lasso_tuned = Lasso(alpha= lasso_cv_model.alpha_)
lasso_tuned.fit(X_train,y_train)
y_pred = lasso_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

#ElasticNet Regresyonu

hit = pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//6)Makine_ogrenmesi/1)Dogrusal_Regresyon/Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

from sklearn.linear_model import ElasticNet
enet_model = ElasticNet().fit(X_train,y_train)

enet_model.coef_
enet_model.intercept_

#Tahmin

enet_model

enet_model.predict(X_test)

y_pred = enet_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
r2_score(y_test,y_pred)

#Model Tuning

from sklearn.linear_model import ElasticNetCV
enet_cv_model = ElasticNetCV(cv = 10, random_state=0).fit(X_train,y_train)
enet_cv_model.alpha_

enet_tuned = ElasticNet(alpha= enet_cv_model.alpha_).fit(X_train,y_train)

y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

