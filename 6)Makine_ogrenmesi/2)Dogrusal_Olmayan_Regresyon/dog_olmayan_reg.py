#Doğrusal Olmayan Regresyon Modelleri
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor

from warnings import filterwarnings
filterwarnings('ignore')

#Data
hit = pd.read_csv("C://Users//Dursun Can//Desktop//VB//Udemy//UdemyDataScienceAndMachineLearning//6)Makine_ogrenmesi/2)Dogrusal_Olmayan_Regresyon/Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

#KNN
knn_model = KNeighborsRegressor()
knn_model.fit(X_train,y_train)

knn_model.n_neighbors

#Tahmin

y_pred = knn_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

RMSE = []

for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors= k).fit(X_train,y_train)
    y_pred = knn_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train,y_pred))
    RMSE.append(rmse)
    print("k =", k , "için RMSE değeri: ", rmse)

#Model Tuning

knn_params = {'n_neighbors': np.arange(1,30,1)}
knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn, knn_params, cv=10)
knn_cv_model.fit(X_train,y_train)
knn_cv_model.best_params_["n_neighbors"]

RMSE = []
RMSE_CV = []

for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors= k).fit(X_train,y_train)
    y_pred = knn_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train,y_pred))
    rmse_cv = np.sqrt(-1*cross_val_score(knn_model, X_train, y_train, cv=10,
                                         scoring= "neg_mean_squared_error").mean())
    RMSE.append(rmse)
    RMSE_CV.append(rmse_cv)
    print("k =", k , "için RMSE değeri: ", rmse, "RMSE_CV değeri: ", rmse_cv)

knn_tuned = KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
knn_tuned.fit(X_train,y_train)
tuned_pred = knn_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,tuned_pred))

#SVR (Destek Vektör Regresyonu)
X_train = pd.DataFrame(X_train["Hits"])
X_test = pd.DataFrame(X_test["Hits"])

from sklearn.svm import SVR
svr_model = SVR("linear")
svr_model.fit(X_train,y_train)

svr_model.predict(X_train)[0:5]

print("y = {0} + {1} x".format(svr_model.intercept_[0],
                               svr_model.coef_[0][0]))
X_train["Hits"][0:1]

-48.69756097561513 + 4.969512195122093 *91

y_pred = svr_model.predict(X_train)
plt.scatter(X_train, y_train)
plt.plot(X_train, y_pred, color ="r")
plt.show()

from sklearn.linear_model import LinearRegression
lm_model = LinearRegression()
lm_model.fit(X_train,y_train)
lm_pred = lm_model.predict(X_train)
print("y = {0} + {1} x".format(lm_model.intercept_, lm_model.coef_[0]))
-8.814095480334572 + 5.1724561354706875 *91

plt.scatter(X_train, y_train, alpha=0.5, s=23)
plt.plot(X_train, lm_pred, 'g')
plt.plot(X_train,y_pred, color= 'r')

plt.xlabel("Atış Sayısı(Hits)")
plt.ylabel("Maaş (Salary)")
plt.show()

#Tahmin
print("y = {0} + {1} x".format(svr_model.intercept_[0], svr_model.coef_[0][0]))

svr_model.predict([[91]])

y_pred = svr_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#Model Tuning

svr_params = {"C": np.arange(0.1,2,0.1)}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv=10)
svr_cv_model.fit(X_train,y_train)
pd.Series(svr_cv_model.best_params_)[0]
svr_tuned = SVR("linear", C= pd.Series(svr_cv_model.best_params_)[0])
svr_tuned.fit(X_train,y_train)

y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Doğrusal Olmayan SVR

np.random.seed(3)

x_sim = np.random.uniform(2, 10, 145)
y_sim = np.sin(x_sim) + np.random.normal(0, 0.4, 145)

x_outliers = np.arange(2.5, 5, 0.5)
y_outliers = -5*np.ones(5)

x_sim_idx = np.argsort(np.concatenate([x_sim, x_outliers]))
x_sim = np.concatenate([x_sim, x_outliers])[x_sim_idx]
y_sim = np.concatenate([y_sim, y_outliers])[x_sim_idx]

from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(np.sin(x_sim[:, np.newaxis]), y_sim)
ols_pred = ols.predict(np.sin(x_sim[:, np.newaxis]))

from sklearn.svm import SVR
eps = 0.1
svr = SVR('rbf', epsilon = eps)
svr.fit(x_sim[:, np.newaxis], y_sim)
svr_pred = svr.predict(x_sim[:, np.newaxis])

plt.scatter(x_sim, y_sim, alpha=0.5, s=26)
plt_ols, = plt.plot(x_sim, ols_pred, 'g')
plt_svr, = plt.plot(x_sim, svr_pred, color='r')
plt.xlabel("Bağımsız Değişken")
plt.ylabel("Bağımlı Değişken")
plt.ylim(-5.2, 2.2)
plt.legend([plt_ols, plt_svr], ['EKK', 'SVR'], loc = 4);
plt.show()

svr_rbf = SVR("rbf")
svr_rbf.fit(X_train,y_train)

#Tahmin

y_pred = svr_rbf.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning

svr_params = {"C": [0.1,0.4,5,10,20,30,40,50]}
svr_cv_model = GridSearchCV(svr_rbf,svr_params,cv=10)
svr_cv_model.fit(X_train,y_train)
svr_cv_model.best_params_
pd.Series(svr_cv_model.best_params_)[0]
svr_tuned = SVR("rbf", C= pd.Series(svr_cv_model.best_params_)[0])
svr_tuned.fit(X_train,y_train)
svr_tuned_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, svr_tuned_pred))

#Çok Katmanlı Algılayıcı

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neural_network import MLPRegressor
mlp_model = MLPRegressor()
mlp_model.fit(X_train_scaled, y_train)

mlp_model.n_layers_

mlp_model.hidden_layer_sizes

mlp_model = MLPRegressor(hidden_layer_sizes= (100,20))
mlp_model.fit(X_train_scaled,y_train)
mlp_model.n_layers_

mlp_model.hidden_layer_sizes

#Tahmin

mlp_model.predict(X_test_scaled)[0:5]

y_pred = mlp_model.predict(X_test_scaled)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning
mlp_params = {'alpha': [0.1,0.01,0.02,0.005],
              'hidden_layer_sizes': [(20,20),(100,50,150),(300,200,150)],
              'activation': ['relu', 'logistic']}

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv = 10)
mlp_cv_model.fit(X_train_scaled, y_train)

mlp_cv_model.best_params_

mlp_tuned = MLPRegressor(alpha= 0.02, hidden_layer_sizes= (100,50,150))
mlp_tuned.fit(X_train_scaled,y_train)
y_pred = mlp_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

#CART

X_train = pd.DataFrame(X_train["Hits"])
X_test = pd.DataFrame(X_test["Hits"])
cart_model = DecisionTreeRegressor()
cart_model

cart_model.fit(X_train,y_train)

X_grid = np.arange(min(np.array(X_train)), max(np.array(X_train)), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, cart_model.predict(X_grid), color = 'blue')
plt.title('CART REGRESYON AĞACI')
plt.xlabel('Atış Sayısı (Hits)')
plt.ylabel('Maaş (Salary');
plt.show()

cart_model = DecisionTreeRegressor(max_leaf_nodes= 3)
cart_model.fit(X_train,y_train)

X_grid = np.arange(min(np.array(X_train)), max(np.array(X_train)), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, cart_model.predict(X_grid), color = 'blue')
plt.title('CART REGRESYON AĞACI')
plt.xlabel('Atış Sayısı (Hits)')
plt.ylabel('Maaş (Salary');
plt.show()

from skompiler import skompile
print(skompile(cart_model.predict).to('python/code'))

#Tahmin

cart_model.predict(X_test)[0:5]

cart_model.predict([[91]])
y_pred = cart_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning

cart_model = DecisionTreeRegressor()
cart_model.fit(X_train,y_train)
y_pred = cart_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

cart_params = {"min_samples_split": range(2,100), "max_leaf_nodes": range(2,10)}
cart_cv_model = GridSearchCV(cart_model, cart_params, cv=10)
cart_cv_model.fit(X_train, y_train)

cart_cv_model.best_params_

cart_tuned = DecisionTreeRegressor(max_leaf_nodes=9, min_samples_split=37)
cart_tuned.fit(X_train, y_train)

y_pred = cart_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#Bagged Trees Regresyon

bag_model = BaggingRegressor(bootstrap_features= True)
bag_model.fit(X_train,y_train)

bag_model.n_estimators
bag_model.estimators_

bag_model.estimators_samples_

bag_model.estimators_features_

bag_model.estimators_[0]

#Tahmin

y_pred = bag_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

iki_y_pred = bag_model.estimators_[1].fit(X_train, y_train).predict(X_test)
np.sqrt(mean_squared_error(y_test, iki_y_pred))

yedi_y_pred = bag_model.estimators_[6].fit(X_train, y_train).predict(X_test)
np.sqrt(mean_squared_error(y_test, iki_y_pred))


#Model Tuning

bag_model = BaggingRegressor(bootstrap_features= True)
bag_model.fit(X_train, y_train)

bag_params = {"n_estimators": range(2,20)}
bag_cv_model = GridSearchCV(bag_model, bag_params, cv=10)
bag_cv_model.fit(X_train, y_train)
bag_cv_model.best_params_

bag_tuned = BaggingRegressor(n_estimators=12, random_state=45)
bag_tuned.fit(X_train,y_train)
y_pred = bag_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Random Forests
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state= 42)
rf_model.fit(X_train, y_train)

#Tahmin

rf_model.predict(X_test)[0:5]
y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

rf_params = {'max_depth': list(range(1,10)),
             'max_features': [3,5,10,15],
             'n_estimators': [200, 500, 1000, 2000]}

rf_model = RandomForestRegressor(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1)

rf_cv_model.fit(X_train, y_train)
rf_cv_model.best_params_

rf_tuned = RandomForestRegressor(max_depth=8, max_features=3, n_estimators=200)
rf_tuned.fit(X_train,y_train)

y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_ * 100},
                          index= X_train.columns)

Importance.sort_values(by = "Importance", axis=0, ascending=True).plot(kind = "barh",
                                                                       color= "r")
plt.xlabel("Değişken Önem Düzeyleri")
plt.show()

#Gradient Boosting Machnies

from sklearn.ensemble import GradientBoostingRegressor
gbm_model = GradientBoostingRegressor()
gbm_model.fit(X_train,y_train)

#Tahmin

y_pred = gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning

gbm_params = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'max_depth': [3,5,8,50,100],
    'n_estimators': [200,500,1000,2000],
    'subsample': [1,0.5,0.75],}

gbm = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm, gbm_params, cv=10, n_jobs=-1, verbose=2)
gbm_cv_model.fit(X_train,y_train)
gbm_cv_model.best_params_
gbm_tuned = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators= 2000,
                                      subsample=0.75)
gbm_tuned =gbm_tuned.fit(X_train,y_train)

y_pred = gbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

Importance = pd.DataFrame({"Importance": gbm_tuned.feature_importances_ * 100},
                          index= X_train.columns)

Importance.sort_values(by = "Importance", axis=0, ascending=True).plot(kind = "barh",
                                                                       color= "r")
plt.xlabel("Değişken Önem Düzeyleri")
plt.show()

import xgboost as xgb
DM_train = xgb.DMatrix(data= X_train, label= y_train)
DM_test = xgb.DMatrix(data= X_test, label= y_test)

from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train,y_train)

#Tahmin

y_pred = xgb.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#Model Tuning

xgb_grid = {'colsample_bytree': [0.4,0.5,0.6,0.9,1],
            'n_estimators': [100,200,500,1000],
            'max_depth': [2,3,4,5,6],
            'learning_rate': [0.1,0.01,0.5]}
xgb = XGBRegressor()

xgb_cv = GridSearchCV(xgb, param_grid= xgb_grid, cv=10, n_jobs=-1, verbose=2)
xgb_cv.fit(X_train,y_train)

xgb_cv.best_params_

xgb_tuned = XGBRegressor(colsample_bytree= 0.5, learning_rate= 0.1, max_depth=2, n_estimators= 500)
xgb_tuned = xgb_tuned.fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Light GBM

from lightgbm import LGBMRegressor

lgbm = LGBMRegressor()
lgbm_model = lgbm.fit(X_train,y_train)

#Tahmin

y_pred = lgbm_model.predict(X_test, num_iteration= lgbm_model.best_iteration_)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning

lgbm_grid = {
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'n_estimators': [20,40,100,200,500,1000],
    'max_depth': [1,2,3,4,5,6,7,8]}

lgbm= LGBMRegressor()
lgbm_cv_model = GridSearchCV(lgbm, lgbm_grid, cv=10, n_jobs=-1, verbose= 2)
lgbm_cv_model.fit(X_train,y_train)

lgbm_cv_model.best_params_

lgbm_tuned = LGBMRegressor(learning_rate=0.1, max_depth=6, n_estimators=20, colsample_bytree=0.6)
lgbm_tuned.fit(X_train,y_train)

y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#CATBOOST

from catboost import CatBoostRegressor
catb = CatBoostRegressor()
catb_model = catb.fit(X_train, y_train)

#Tahmin

y_pred = catb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#Model Tuning

catb_grid = {
    'iterations': [200,500,1000,2000],
    'learning_rate': [0.01,0.03, 0.05, 0.1],
    'depth': [3,4,5,6,7,8]}

catb = CatBoostRegressor()
catb_cv_model = GridSearchCV(catb, catb_grid, cv=5, n_jobs=-1, verbose=2)
catb_cv_model.fit(X_train,y_train)

catb_cv_model.best_params_

catb_tuned = CatBoostRegressor(iterations= 200,
                               learning_rate= 0.01,
                               depth= 8)
catb_tuned =catb_tuned.fit(X_train, y_train)

y_pred = catb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

