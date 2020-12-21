#Veriye İlk Bakış

#Veri Seti Hikayesi ve Yapısının İncelenmesi

import seaborn as sns
planets = sns.load_dataset("planets")
planets.head()

df = planets.copy()

df.head()
df.tail()

#veri seti yapısal bilgileri
df.info()

df.dtypes

import pandas as pd
df.method = pd.Categorical(df.method)

df.dtypes

df.head()

#Veri Setinin Betimlenmesi
import seaborn as sns
planets = sns.load_dataset("planets")
planets.head()
df = planets.copy()
df.shape
df.columns
df.describe().T

df.describe(include= "all").T

#Eksik Değerlerin İncelenmesi

#hiç eksik gözlem var mı?
df.isnull().values.any()

#Hangi değişkende kaçar tane var?
df.isnull().sum()

df["orbital_period"].fillna(0, inplace = True)
df.isnull().sum()

df["mass"].fillna(df.mass.mean(), inplace = True)
df.isnull().sum()

df.fillna(df.mean(), inplace=True)
df.isnull().sum()

df = planets.copy()
df.head()

#Kategorik Değişken Özetleri

import seaborn as sns
planets = sns.load_dataset("planets")
df = planets.copy()
df.head()

#Sadece Kategorik Değişkenler ve Özetleri

kat_df = df.select_dtypes(include= ["object"])
kat_df.head()

#Kategorik Değişkenin Sınıflarına ve Sınıf Sayısına Erişmek

kat_df.method.unique()
kat_df["method"].value_counts().count()

#Kategorik Değişkenin Sınıflarının Frekanslarına Erişmek
kat_df["method"].value_counts()

import matplotlib.pyplot as plt
df["method"].value_counts().plot.barh();
plt.show()

#Süreki Değişken Özetleri

import seaborn as sns
planets = sns.load_dataset("planets")
df = planets.copy()
df.head()

df_num = df.select_dtypes(include= ["float64", "int64"])
df_num.head()
df_num.describe().T

df_num["distance"].describe()

print("Ortalama"+ str(df_num["distance"].mean()))
print("Dolu Gözlem Sayısı: "+ str(df_num["distance"].count()))
print("Maksimum Değer: "+ str(df_num["distance"].max()))
print("Minimum Değer: "+ str(df_num["distance"].min()))
print("Medyan: "+ str(df_num["distance"].median()))
print("Standart Sapma: "+ str(df_num["distance"].std()))

#Dağılım Grafikleri
#Barplot

import  seaborn as sns
diamonds = sns.load_dataset("diamonds")
df = diamonds.copy()
df.head()

#Veri Setine Hızlı Bakış

df.info()

df.describe().T

df["cut"].value_counts()

df["color"].value_counts()

#ordinal tanımlama
from pandas.api.types import CategoricalDtype
df.cut.head()
df.cut = df.cut.astype(CategoricalDtype(ordered= True))
df.dtypes
df.cut.head(1)

cut_kategoriler = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
df.cut = df.cut.astype(CategoricalDtype(categories= cut_kategoriler, ordered= True))
df.cut.head(1)

#barplot
import matplotlib.pyplot as plt
df["cut"].value_counts().plot.barh().set_title("Cut Değişkeninin Sınıf Frekansları");
plt.show()
(df["cut"]
 .value_counts()
 .plot.barh()
 .set_title("Cut Değişkeninin Sınıf Frekansları"));
plt.show()

sns.barplot(x = "cut", y= df.cut.index, data=df);
plt.show()

#Çarprazlamalar
import matplotlib.pyplot as plt
import  seaborn as sns
from pandas.api.types import CategoricalDtype
diamonds = sns.load_dataset("diamonds")
df = diamonds.copy()
cut_kategoriler = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
df.cut = df.cut.astype(CategoricalDtype(categories= cut_kategoriler, ordered= True))
df.head()

sns.catplot(x = "cut", y= "price", data=df)
plt.show()

sns.barplot(x="cut", y="price", hue="color", data=df);
plt.show()

df.groupby(["cut", "color"]) ["price"].mean()

#Histogram ve Yoğunluk

sns.distplot(df.price, kde= False);
plt.show()

sns.distplot(df.price, bins=10, kde= False);
plt.show()

sns.distplot(df.price);
plt.show()

sns.distplot(df.price, hist = False);
plt.show()

df["price"].describe()

sns.kdeplot(df.price, shade= True);
plt.show()

#Çarprazlamalar

sns.kdeplot(df.price, shade= True);
plt.show()

(sns.FacetGrid(df, hue= "cut", height=5, xlim= (0,1000)).map(sns.kdeplot, "price", shade = True).add_legend());
plt.show()

sns.catplot(x= "cut", y="price", hue="color", kind="point", data=df);
plt.show()

#Boxplot

import seaborn as sns
import matplotlib.pyplot as plt
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()

df.describe().T

df["sex"].value_counts()
df["smoker"].value_counts()
df["day"].value_counts()
df["time"].value_counts()

#Boxplot

sns.boxplot(x= df["total_bill"]);
plt.show()

sns.boxplot(x= df["total_bill"], orient="v");
plt.show()

#Çaprazlamalar

df.describe().T

#Hangi Günler daha fazla kazanıyoruz?
sns.boxplot(x = "day",y = "total_bill", data=df)
plt.show()

#sabah mı akşam mı daha çok kazanıyoruz?
sns.boxplot(x = "time",y = "total_bill", data=df)
plt.show()

#kişi sayısı kazanç

sns.boxplot(x = "size",y = "total_bill", data=df)
plt.show()

sns.boxplot(x = "day",y = "total_bill", hue="sex", data=df)
plt.show()

#Violin
sns.catplot(y= "total_bill", kind="violin", data=df);
plt.show()

#Çarprazlamalar

sns.catplot(x="day", y= "total_bill", kind="violin", data=df);
plt.show()

sns.catplot(x="day", y= "total_bill", hue="sex", kind="violin", data=df);
plt.show()

#Korelasyon Grafikleri
#Scatterplot

import seaborn as sns
import matplotlib.pyplot as plt
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()

sns.scatterplot(x = "total_bill", y= "tip", data=df);
plt.show()

#Çaprazlamalar

sns.scatterplot(x = "total_bill", y= "tip", hue="time", data=df);
plt.show()

sns.scatterplot(x = "total_bill", y= "tip", hue="time", style="day", data=df);
plt.show()

sns.scatterplot(x = "total_bill", y= "tip",hue="size", size="size", data=df);
plt.show()

#Doğrusal İlişkinin Gösterilmesi
import seaborn as sns
import matplotlib.pyplot as plt
tips = sns.load_dataset("tips")
df = tips.copy()

sns.lmplot(x= "total_bill", y="tip", data=df);
plt.show()

sns.lmplot(x= "total_bill", y="tip", hue="smoker", data=df);
plt.show()

sns.lmplot(x= "total_bill", y="tip", hue="smoker", col="time", data=df);
plt.show()

sns.lmplot(x= "total_bill", y="tip", hue="smoker", col="time", row="sex", data=df);
plt.show()

#Scatterplot Matrisi
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset("iris")
df = iris.copy()
df.head()

df.dtypes
df.shape

sns.pairplot(df);
plt.show()

sns.pairplot(df, hue="species");
plt.show()

sns.pairplot(df, hue="species", markers= ["o","s","D"]);
plt.show()

sns.pairplot(df, kind="reg");
plt.show()

#Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
flights = sns.load_dataset("flights")
df = flights.copy()
df.head()

df.shape
df["passengers"].describe()

df = df.pivot("month","year","passengers");
df
sns.heatmap(df);
plt.show()

sns.heatmap(df, annot = True, fmt="d");
plt.show()

sns.heatmap(df, annot = True, fmt="d", linewidths=.5);
plt.show()

sns.heatmap(df, annot = True, fmt="d", linewidths=.5,cbar =False);
plt.show()

#Çizgi Grafik

import seaborn as sns
import matplotlib.pyplot as plt
fmri = sns.load_dataset("fmri")
df = fmri.copy()
df.head()

df.shape

df["timepoint"].describe()
df["signal"].describe()

df.groupby("timepoint")["signal"].count()

df.groupby("signal")["timepoint"].count()

df.groupby("signal").count()

df.groupby("timepoint")["signal"].describe()

#Çizgi grafik ve çarprazlamalar

sns.lineplot(x = "timepoint", y= "signal", data=df)
plt.show()

sns.lineplot(x = "timepoint", y= "signal", hue= "event", data=df)
plt.show()

sns.lineplot(x = "timepoint", y= "signal", hue= "event", style= "event", data=df);
plt.show()

sns.lineplot(x = "timepoint", y= "signal", hue= "event", style= "event", markers=True,dashes=False, data=df);
plt.show()

sns.lineplot(x = "timepoint", y= "signal", hue= "event", style= "event", markers=True,dashes=False, data=df);
plt.show()
sns.lineplot(x = "timepoint", y= "signal", hue= "region", style= "event", data=df);
plt.show()

#Basit zaman Serisi Grafiği
import pandas_datareader as pr
import pandas as pd
import matplotlib.pyplot as plt
df = pr.get_data_yahoo("AAPL", start ="2016-01-01", end= "2019-08-25")
df.head()

df.shape

kapanis = df["Close"]
kapanis.head()
kapanis.plot();
plt.show()

kapanis.index

kapanis.index = pd.DatetimeIndex(kapanis.index)
kapanis.head()

kapanis.plot();
plt.show()
