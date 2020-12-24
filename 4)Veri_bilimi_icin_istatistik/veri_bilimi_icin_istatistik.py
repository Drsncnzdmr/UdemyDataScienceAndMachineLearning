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

#Büyük Sayılar Yasası

import numpy as np
rng = np.random.RandomState(123)
for i in np.arange(1,21):
    deney_sayisi = 2**i
    yazi_turalar = rng.randint(0, 2, size=deney_sayisi)
    yazi_olasiliklari = np.mean(yazi_turalar)
    print("Atış Sayısı:",deney_sayisi,"---","Yazı Olasılığı: %2.f" % (yazi_olasiliklari))

#Binom Dağılımı
from scipy.stats import binom
p = 0.01
n = 100
rv = binom(n, p)
print(rv.pmf(1))
print(rv.pmf(5))
print(rv.pmf(10))

#Poisson Dağılımı
from scipy.stats import poisson
lambda_ = 0.1
rv = poisson(mu = lambda_)
print(rv.pmf(k = 0))
print(rv.pmf(k = 3))
print(rv.pmf(k = 5))

#Normal Dağılım
from scipy.stats import norm


#90'dan fazla olması
1-norm.cdf(90, 80, 5)

#70'den fazla olması
1-norm.cdf(70, 80, 5)

#73'den az olması
norm.cdf(73, 80, 5)

#85 ile 90 arasında olması
norm.cdf(90, 80, 5) - norm.cdf(85, 80, 5)

#Tek Örneklem T Testi
#Örnek ortalamasına ilişkin test yapmak için kullanılır.

import numpy as np
olcumler = np.array([17,160,234,149,145,107,197,75,201,225,211,119,
                     157,145,127,244,163,114,145,65,112,185,202,146,
                     203,224,203,114,188,156,187,154,177,95,165,50,110,
                     216,138,151,166,135,155,84,251,173,131,207,121,120])
olcumler[0:10]
import scipy.stats as stast
stast.describe(olcumler)

#Varsayımlar
#Normallik varsayımı

import pandas as pd
import matplotlib.pyplot as plt
#Histogram
pd.DataFrame(olcumler).plot.hist();
plt.show()

#qqplot
import pylab
stast.probplot(olcumler, dist="norm", plot=pylab)
pylab.show()

#Shapiro-Wilks Testi

#H0: Örnek dağılımı ile teorik normal dağılım arasında istatistiksel olarak anlamlı bir farklılık yoktur.
#H1: Fark vardır.

from scipy.stats import shapiro
shapiro(olcumler)

print("T Hesap İstatistiği:"+ str(shapiro(olcumler)[0]))
print("Hesaplanan P-value:"+ str(shapiro(olcumler)[1]))

#Hipotez Testinin Uygulanması
stast.ttest_1samp(olcumler, popmean=170)

#H0: Web sitemizde geçirilen ortalama süre 170'dir.
#H1: ... değildir.

#H0 reddedilir. p-value < 0.05

#Nonparametrik Tek Örneklem Testi
from statsmodels.stats.descriptivestats import sign_test
sign_test(olcumler,170)

#Tek Örneklem Oran Testi
#H0: p = 0.125
#H1: p!= 0.125

from statsmodels.stats.proportion import proportions_ztest
count = 40
nobs = 500
value = 0.125

proportions_ztest(count, nobs, value)

#H0 reddedilir.

#Bağımsı İki Örneklem T Testi
#H0: M1 = M2
#H1: M1 != M2

#Veri Tipi 1
import pandas as pd
import numpy as np
A = pd.DataFrame([30,21,21,27,29,30,20,20,27,32,35,22,24,23,25,27,23,27,23,25,21,18,24,26,27,28,19,25])

B = pd.DataFrame([37,39,31,31,34,38,30,36,29,28,38,28,37,37,30,32,31,31,27,32,33,33,33,31,32,33,26,32])

A_B = pd.concat([A,B], axis=1)
A_B.columns = ["A", "B"]
A_B.head()

#Veri Tipi 2

A = pd.DataFrame([30,21,21,27,29,30,20,20,27,32,35,22,24,23,25,27,23,27,23,25,21,18,24,26,27,28,19,25])

B = pd.DataFrame([37,39,31,31,34,38,30,36,29,28,38,28,37,37,30,32,31,31,27,32,33,33,33,31,32,33,26,32])
#A ve A'nın grubu
GRUP_A = np.arange(len(A))
GRUP_A = pd.DataFrame(GRUP_A)
GRUP_A[:] = "A"
A = pd.concat([A, GRUP_A], axis=1)

#B ve B'nin grubu
GRUP_B = np.arange(len(B))
GRUP_B = pd.DataFrame(GRUP_B)
GRUP_B[:] = "B"
B = pd.concat([B, GRUP_B], axis=1)

#Tum veri
AB = pd.concat([A,B])
AB.columns = ["gelir", "GRUP"]
print(AB.head())
print(AB.tail())

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x = "GRUP", y="gelir",data=AB);
plt.show()

#Varsayım Kontrolü
A_B.head()
AB.head()

#Normallik Varsayımı
from scipy import stats
from scipy.stats import shapiro
shapiro(A_B.A)
shapiro(A_B.B)

#varyans homojenligi varsayımı
#H0: varyanslar homojendir
#H1: varyansalar homojen değildir.

from scipy.stats import levene
stats.levene(A_B.A, A_B.B)

#Hipotez Testi
stats.ttest_ind(A_B["A"], A_B["B"], equal_var= True)

test_istatistigi, pvalue =stats.ttest_ind(A_B["A"], A_B["B"], equal_var=True)
print("Test_istatistigi = %.4f, p-değeri = %.4f" % (test_istatistigi,pvalue))

#Nonparametrik Bağımsız İki Örneklem Testi
stats.mannwhitneyu(A_B["A"], A_B["B"])
test_istatistigi, pvalue =stats.mannwhitneyu(A_B["A"], A_B["B"])
print("Test_istatistigi = %.4f, p-değeri = %.4f" % (test_istatistigi,pvalue))


#Bağımlı İki Örneklem T Testi

oncesi = pd.DataFrame([123,119,119,116,123,123,121,120,117,118,121,121,123,119,
                       121,118,124,121,125,115,115,119,118,121,117,117,120,120,
                       121,117,118,117,123,118,124,121,115,118,125,115])
sonrasi = pd.DataFrame([118,127,122,132,129,123,129,132,128,130,128,138,140,130,
                        134,134,124,140,134,129,129,138,134,124,122,126,133,127,
                        130,130,130,132,117,130,125,129,133,120,127,123])

#birinci veri seti
AYRIK = pd.concat([oncesi,sonrasi], axis=1)
AYRIK.columns = ["ONCESI", "SONRASI"]
print("'AYRIK' Veri Seti: \n\n", AYRIK.head(),"\n\n")

#İkinci Veri seti
#Oncesi FLAG/TAG'ını oluşturma

GRUP_ONCESI = np.arange(len(oncesi))
GRUP_ONCESI = pd.DataFrame(GRUP_ONCESI)
GRUP_ONCESI[:] = "ONCESI"
#FLAG VE ONCESI DEGERLERINI BIR ARAYA GETIRME
A = pd.concat([oncesi, GRUP_ONCESI], axis=1)
#Sonrası FLAG/TAG'ını oluşturma
GRUP_SONRASI = np.arange(len(sonrasi))
GRUP_SONRASI = pd.DataFrame(GRUP_SONRASI)
GRUP_SONRASI[:] = "SONRASI"

#FLAG VE SONRASI DEGERLERİNİ BİR ARAYA GETİRME
B = pd.concat([sonrasi, GRUP_SONRASI], axis=1)

#TUM VERİYİ BİR ARAYA GETİRME
BIRLIKTE = pd.concat([A,B])
BIRLIKTE

#ISIMLENDIRME
BIRLIKTE.columns = ["PERFORMANS","ONCESI_SONRASI"]
print("'BIRLIKTE' Veri Seti: \n\n", BIRLIKTE.head(), "\n")

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x = "ONCESI_SONRASI", y="PERFORMANS", data=BIRLIKTE);
plt.show()

#Varsayım Kontrolleri
from scipy.stats import shapiro

shapiro(AYRIK.ONCESI)
shapiro(AYRIK.SONRASI)

import scipy.stats as stats
stats.levene(AYRIK.ONCESI, AYRIK.SONRASI)

#Hipotez Testi
stats.ttest_rel(AYRIK.ONCESI, AYRIK.SONRASI)

test_istatistigi, pvalue =stats.ttest_rel(AYRIK["ONCESI"], AYRIK["SONRASI"])
print("Test_istatistigi = %.5f, p-değeri = %.5f" % (test_istatistigi,pvalue))

#Nonparametrik Bağımlı İki Örneklem Testi

stats.wilcoxon(AYRIK.ONCESI, AYRIK.SONRASI)

test_istatistigi, pvalue =stats.wilcoxon(AYRIK["ONCESI"], AYRIK["SONRASI"])
print("Test_istatistigi = %.4f, p-değeri = %.4f" % (test_istatistigi,pvalue))

#İki Örneklem Oran Testi
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
basari_sayisi = np.array([300,250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count= basari_sayisi, nobs= gozlem_sayilari)

#Varyans Analizi

#H0: M1=M2=M3(grup ortalamaları arasında istatiksel anlamda farklılık yoktur).
#H1: Fark vardır.
import pandas as pd
A = pd.DataFrame([28,33,30,29,28,29,27,31,30,32,28,33,25,29,27,31,31,30,31,34,30,32,31,34,28,32,31,28,33,29])

B = pd.DataFrame([31,32,30,30,33,32,34,27,36,30,31,30,38,29,30,34,34,31,35,35,33,30,28,29,26,37,31,28,34,33])

C = pd.DataFrame([40,33,38,41,42,43,38,35,39,39,36,34,35,40,38,36,39,36,33,35,38,35,40,40,39,38,38,43,40,42])

dfs = [A, B, C]

ABC = pd.concat(dfs, axis = 1)
ABC.columns = ["GRUP_A","GRUP_B","GRUP_C"]
ABC.head()

#Varsayım Kontrolü
from scipy.stats import shapiro
shapiro(ABC["GRUP_A"])
shapiro(ABC["GRUP_B"])
shapiro(ABC["GRUP_C"])

import scipy.stats as stats
stats.levene(ABC["GRUP_A"], ABC["GRUP_B"],ABC["GRUP_C"])

#Hipotez Testi

from scipy.stats import f_oneway
f_oneway(ABC["GRUP_A"], ABC["GRUP_B"],ABC["GRUP_C"])
print('{:.5f}'.format(f_oneway(ABC["GRUP_A"], ABC["GRUP_B"],ABC["GRUP_C"])[1]))
ABC.describe().T

#Nonparametrik Hipotez Testi
from scipy.stats import kruskal
kruskal(ABC["GRUP_A"], ABC["GRUP_B"],ABC["GRUP_C"])

#Korelasyon Analizi

import seaborn as sns
tips = sns.load_dataset('tips')
df = tips.copy()
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]
df.head()
df.plot.scatter("tip","total_bill");
import matplotlib.pyplot as plt
plt.show()

#Varsayım Kontrolü

from scipy.stats import shapiro

test_istatistigi, pvalue = shapiro(df["tip"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

test_istatistigi, pvalue = shapiro(df["total_bill"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

#Hipotez Testi

#Korelasyon Katsayısı
df["tip"].corr(df["total_bill"])
df["tip"].corr(df["total_bill"], method = "spearman")

#Korelasyonunu Anlamlılığının Testi
from scipy.stats.stats import pearsonr

test_istatistigi, pvalue = pearsonr(df["tip"],df["total_bill"])
print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

#Nonparametrik Hipotez Testi

from scipy.stats import stats
stats.spearmanr(df["tip"],df["total_bill"])

test_istatistigi, pvalue = stats.spearmanr(df["tip"],df["total_bill"])
print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

test_istatistigi, pvalue = stats.kendalltau(df["tip"],df["total_bill"])
print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
