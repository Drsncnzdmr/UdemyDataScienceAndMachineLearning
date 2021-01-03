#Sınıflandırma Problemleri
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')

diabetes = pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//6)Makine_ogrenmesi/3)Siniflandırma/original.csv")
df = diabetes.copy()
df = df.dropna()
df.head()

df.info()

df["Outcome"].value_counts()

df["Outcome"].value_counts().plot.barh();
plt.show()

df.describe().T

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

loj = sm.Logit(y,X)
loj_model = loj.fit()
loj_model.summary()

#scikit-learn
from sklearn.linear_model import LogisticRegression
loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(X,y)
loj_model

loj_model.intercept_
loj_model.coef_

#Tahmin & Model Tuning

y_pred = loj_model.predict(X)
confusion_matrix(y, y_pred)

accuracy_score(y,y_pred)
print(classification_report(y,y_pred))

loj_model.predict(X)[0:5]

loj_model.predict_proba(X)[0:10]

y_probs = loj_model.predict_proba(X)
y_probs = y_probs[:,1]
y_probs[0:10]

y_pred = [1 if i >0.5 else 0 for i in y_probs]
y_pred[0:10]

confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)

print(classification_report(y,y_pred))

logit_roc_auc = roc_auc_score(y, loj_model.predict(X))

fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label = 'AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

loj = LogisticRegression(solver='liblinear')
loj_model = loj.fit(X_train,y_train)
loj_model

accuracy_score(y_test, loj_model.predict(X_test))

cross_val_score(loj_model, X_test, y_test, cv=10).mean()

#Gaussian NAive Bayes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_model = nb
nb_model.fit(X_train,y_train)

nb_model.predict(X_test)[0:10]

nb_model.predict_proba(X_test)[0:10]

y_pred = nb_model.predict(X_test)

accuracy_score(y_test,y_pred)

cross_val_score(nb_model, X_test,y_test, cv=10).mean()

#KNN

knn = KNeighborsClassifier()
knn_model = knn
knn_model.fit(X_train,y_train)

y_pred = knn_model.predict(X_test)
accuracy_score(y_test,y_pred)

print(classification_report(y_test, y_pred))

#Model Tuning

knn_params = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,knn_params,cv=10)
knn_cv.fit(X_train,y_train)
print("En iyi skor:" + str(knn_cv.best_score_))
print("En iyi parametreler:" +str(knn_cv.best_params_))

knn = KNeighborsClassifier(11)
knn_tuned = knn.fit(X_train,y_train)

knn_tuned.score(X_test, y_test)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

#SVC

svc_model = SVC(kernel="linear")
svc_model.fit(X_train,y_train)

y_pred = svc_model.predict(X_test)
accuracy_score(y_test, y_pred)

#Model Tuning

svc_params = {"C": np.arange(1,10)}

svc = SVC(kernel="linear")

svc_cv_model = GridSearchCV(svc, svc_params, cv=10, n_jobs=-1, verbose=2)
svc_cv_model.fit(X_train,y_train)

print("En iyi parametreler:" +str(svc_cv_model.best_params_))

svc_tuned = SVC(kernel="linear", C=5)
svc_tuned.fit(X_train,y_train)

y_pred = svc_tuned.predict(X_test)
accuracy_score(y_test,y_pred)

#RBF SVC

svc_model = SVC(kernel="rbf")
svc_model.fit(X_train,y_train)

y_pred = svc_model.predict(X_test)
accuracy_score(y_test,y_pred)

#Model Tuning

svc_params = {'C': [0.0001, 0.001, 0.1, 1,5,10,50,100],
              "gamma":[0.0001, 0.001, 0.1, 1,5,10,50,100]}

svc = SVC()
svc_cv_model = GridSearchCV(svc, svc_params,cv=10,n_jobs=-1, verbose=2)
svc_cv_model.fit(X_train,y_train)

print("En iyi parametreler:" +str(svc_cv_model.best_params_))

svc_tuned = SVC(C =10, gamma=0.0001)
svc_tuned.fit(X_train,y_train)

y_pred = svc_tuned.predict(X_test)
accuracy_score(y_test,y_pred)

#Yapay Sinir Ağları

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled[0:5]

from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier()
mlpc.fit(X_train_scaled, y_train)

mlpc.coefs_
y_pred = mlpc.predict(X_test_scaled)
accuracy_score(y_test,y_pred)

#Model Tuning

mlpc_params = {"alpha": [0.1,0.01,0.02,0.005,0.0001,0.00001],
               "hidden_layer_sizes": [(10,10,10),
                                      (100,100,100),
                                      (100,100),
                                      (3,5),
                                      (5,3)],
               "solver": ["lbfgs", "adam", "sgd"],
               "activation": ["relu", "logistic"]}
mlpc = MLPClassifier()
mlpc_cv_model = GridSearchCV(mlpc, mlpc_params, cv=10, n_jobs=-1, verbose=2)
mlpc_cv_model.fit(X_train_scaled, y_train)
print("En iyi parametreler:" +str(mlpc_cv_model.best_params_))

mlpc_tuned = MLPClassifier(activation="relu", alpha=0.01, hidden_layer_sizes=(100,100,100), solver="sgd")
mlpc_tuned.fit(X_train_scaled,y_train)

y_pred = mlpc_tuned.predict(X_test_scaled)
accuracy_score(y_test,y_pred)

#CART

from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
cart.fit(X_train,y_train)

y = df["Outcome"]
X = df["Pregnancies"]
X= pd.DataFrame(X)
from skompiler import skompile

print(skompile(cart.predict).to("python/code"))

x = [9]
((0 if x[0] <= 2.5 else 0) if x[0] <= 6.5 else 1 if x[0] <= 13.5 else 1)

y_pred = cart.predict(X_test)
accuracy_score(y_test,y_pred)

#Model Tuning

cart_grid = {'max_depth': list(range(1,10)),
             'min_samples_split':list(range(2,50)) }

cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv=10, n_jobs=-1, verbose=2)
cart_cv_model = cart_cv.fit(X_train,y_train)

print("En iyi parametreler:" +str(cart_cv_model.best_params_))

cart_tuned = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=2)
cart_tuned.fit(X_train,y_train)

y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test,y_pred)

#Random Forests

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)

y_pred = rf_model.predict(X_test)
accuracy_score(y_test,y_pred)

rf_params = {"max_depth": [2,5,8,10],
             "max_features": [2,5,8],
             "n_estimators": [10,500,1000],
             "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()
rf_cv_model = GridSearchCV(rf_model,rf_params, cv=10, n_jobs=-1,verbose=2)
rf_cv_model.fit(X_train,y_train)

rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,5,8],
            "n_estimators": [10,500,1000],
            "min_samples_split": [2,5,10]}

rf_model = RandomForestClassifier()

rf_cv_model = GridSearchCV(rf_model,
                           rf_params,
                           cv = 10,
                           n_jobs = -1,
                           verbose = 2)
rf_cv_model.fit(X_train, y_train)
print("En iyi parametreler:" +str(rf_cv_model.best_params_))


rf_tuned = RandomForestClassifier(max_depth=10,
                                  max_features = 8,
                                  min_samples_split=10,
                                  n_estimators=1000)
rf_tuned.fit(X_train,y_train)

y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test,y_pred)


Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                          index= X_train.columns)
Importance.sort_values(by="Importance",
                       axis=0, ascending=True).plot(kind = "barh", color="r")
plt.xlabel("Değişken Önem Düzeyleri")
plt.show()

#Gradient Boosting Machines

from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X_train,y_train)

y_pred = gbm_model.predict(X_test)
accuracy_score(y_test,y_pred)

#Model tuning

gbm_params = {"learning_rate": [0.001, 0.01, 0.01, 0.1, 0.05],
              "n_estimators": [100,500,100],
              "max_depth": [3,5,10],
              "min_samples_split": [2,5,10]}
gbm = GradientBoostingClassifier()
gbm_cv = GridSearchCV(gbm, gbm_params, cv=10, n_jobs=-1, verbose=2)
gbm_cv.fit(X_train,y_train)

print("En iyi parametreler:" +str(gbm_cv.best_params_))

gbm = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, min_samples_split=5, n_estimators=100)
gbm_tuned = gbm.fit(X_train,y_train)

y_pred = gbm_tuned.predict(X_test)
accuracy_score(y_test,y_pred)

#XGBOOST
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train,y_train)

y_pred= xgb_model.predict(X_test)
accuracy_score(y_test,y_pred)

#Model Tuning

xgb_params = {'n_estimators': [100,500,1000,2000],
              'subsample': [0.6,0.8,1.0],
              'max_depth': [3,4,5,6],
              'learning_rate': [0.1,0.01,0.02,0.05],
              'min_samples_split': [2,5,10]}
xgb = XGBClassifier()
xgb_cv_model = GridSearchCV(xgb, xgb_params, cv=10, n_jobs=-1, verbose=2)
xgb_cv_model.fit(X_train,y_train)

xgb_cv_model.best_params_

xgb_tuned = XGBClassifier(learning_rate=0.01,max_depth=6, min_samples_split = 2,n_estimators=100,subsample=0.8)
xgb_tuned.fit(X_train,y_train)

y_pred = xgb_tuned.predict(X_test)
accuracy_score(y_test,y_pred)

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
lgbm.fit(X_train,y_train)

y_pred = lgbm.predict(X_test)
accuracy_score(y_test,y_pred)

#Model Tuning

lgbm_params = {'n_estimators': [100,500,1000,2000],
               'subsample': [0.6, 0.8, 1.0],
               'max_depth': [3,4,5,6],
               'learning_rate': [0.1,0.01,0.02,0.05],
               'min_child_split': [5,10,20]}
lgbm = LGBMClassifier()
lgbm_cv_model = GridSearchCV(lgbm,lgbm_params,cv=10, n_jobs=-1, verbose=2)
lgbm_cv_model.fit(X_train,y_train)

lgbm_cv_model.best_params_

lgbm_tuned = LGBMClassifier(learning_rate=0.01, max_depth=3, subsample=0.6, n_estimators=500, min_child_samples=20)
lgbm_tuned.fit(X_train,y_train)

y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

#CatBoost

from catboost import CatBoostClassifier

catb_model = CatBoostClassifier()
catb_model.fit(X_train,y_train)

y_pred = catb_model.predict(X_test)
accuracy_score(y_test,y_pred)

#Model Tuning

catb_params = {'iterations': [200,500],
               'learning_rate': [0.01,0.05, 0.1],
               'depth': [3,5,8]}

catb = CatBoostClassifier()
catb_cv_model = GridSearchCV(catb, catb_params, cv=5, n_jobs=-1, verbose=2)
catb_cv_model.fit(X_train,y_train)
catb_cv_model.best_params_
catb_tuned = CatBoostClassifier(iterations = 200, learning_rate = 0.05, depth = 5)

y_pred = catb_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

#Tüm Modellerin Karşılaştırılması

modeller = [ knn_tuned,loj_model,svc_tuned,nb_model, mlpc_tuned,cart_tuned, rf_tuned,gbm_tuned,catb_tuned, lgbm_tuned,xgb_tuned]

for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_pred)
    print("-" * 28)
    print(isimler + ":")
    print("Accuracy: {:.4%}".format(dogruluk))

sonuc = []

sonuclar = pd.DataFrame(columns=["Modeller", "Accuracy"])

for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_pred)
    sonuc = pd.DataFrame([[isimler, dogruluk * 100]], columns=["Modeller", "Accuracy"])
    sonuclar = sonuclar.append(sonuc)

sns.barplot(x='Accuracy', y='Modeller', data=sonuclar, color="r")
plt.xlabel('Accuracy %')
plt.title('Modellerin Doğruluk Oranları');
