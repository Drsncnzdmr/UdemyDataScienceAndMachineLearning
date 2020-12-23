#Veri Bilimi İçin İstatistik

#Örnek Teorisi

import numpy as np
populasyon = np.random.randint(0,80,10000)
populasyon[0:10]

#Örneklem çekimi

np.random.seed(115)

orneklem = np.random.choice(a = populasyon, size=100)
orneklem[0:10]

orneklem.mean()
populasyon.mean()

#örneklem dağılımı

np.random.seed(10)
orneklem1 = np.random.choice(a = populasyon, size=100)
orneklem2 = np.random.choice(a = populasyon, size=100)
orneklem3 = np.random.choice(a = populasyon, size=100)
orneklem4 = np.random.choice(a = populasyon, size=100)
orneklem5 = np.random.choice(a = populasyon, size=100)
orneklem6 = np.random.choice(a = populasyon, size=100)
orneklem7 = np.random.choice(a = populasyon, size=100)
orneklem8 = np.random.choice(a = populasyon, size=100)
orneklem9 = np.random.choice(a = populasyon, size=100)
orneklem10 = np.random.choice(a = populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean()+ orneklem4.mean() + orneklem5.mean()+
orneklem6.mean() + orneklem7.mean() + orneklem8.mean() +orneklem9.mean() + orneklem10.mean()) / 10

orneklem1.mean()
orneklem3.mean()
orneklem9.mean()

#Betimsel İstatistikler

import seaborn as sns
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()

df.describe().T

import researchpy as rp

rp.summary_cont(df[["total_bill", "tip", "size"]])

rp.summary_cat(df[["sex", "smoker", "day"]])

df[["tip","total_bill"]].cov()

df[["tip","total_bill"]].corr()

#İş Uygulamaso: Fiyat Stratejisi Karar Destek
import numpy as np
fiyatlar = np.random.randint(10, 110, 1000)
fiyatlar.mean()

import statsmodels.stats.api as sms
sms.DescrStatsW(fiyatlar).tconfint_mean()
#Çıktı = Kullanıcıların ödeyebileceği ücret aralığını gösterir.

#Olasılık Dağılımları
#Bernoulli Dağılımı

from scipy.stats import bernoulli
p = 0.6 #Tura gelme oalsılığı %60

rv = bernoulli(p)
rv.pmf(k = 1)

