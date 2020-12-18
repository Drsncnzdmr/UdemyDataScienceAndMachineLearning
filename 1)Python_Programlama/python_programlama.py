#STRING METODLARI - LEN

"""
mvk = "geleceği_yazanlar" #Atama işlemleri

a = 9
b = 10

a*b

print(mvk)
print(len(mvk))
"""

#STRING METODLARI - UPPER & LOWER
"""
mvk = "geleceği_yazanlar"

mvk_upper = mvk.upper()
print(mvk_upper)

mvk_lower = mvk.lower()
print(mvk_lower)

print(mvk.islower())
print(mvk_upper.islower())
print(mvk_upper.isupper())
"""

#STRING METODLARI - Replace()
"""
#Yerine koymak için kullanılır.

gel_yaz = "geleceği_yazanlar"

gel_yaz_rep = gel_yaz.replace("e", "a") #e harflerini a yapar.
print(gel_yaz)
print(gel_yaz_rep)
"""

#STRING METODLARI - Strip()
#Kırpma işlemi sağlar.
"""
gel_yaz = " geleceği_yazanlar "

gel_yaz_striped = gel_yaz.strip() #Boşlukları sildi.

print(gel_yaz)
print(gel_yaz_striped)

gel_yaz2 = "*geleceği yazanlar*"
gel_yaz_striped2 = gel_yaz2.strip("*")
print(gel_yaz2)
print(gel_yaz_striped2)
"""

#METODLARA GENEL BAKIŞ
"""
gel_yaz = "geleceği yazanlar"

gel_yaz_cap = gel_yaz.capitalize() #Baş harfi büyütür.
print(gel_yaz_cap)

gel_yaz_title = gel_yaz.title() #Her başlangıçı büyüttü.
print(gel_yaz_title)
"""

#Substrings
"""
gel_yaz = "geleceği_yazanlar"

print(gel_yaz[0])
print(gel_yaz[0:3])
print(gel_yaz[3:7])
"""

#Değişkenler
"""
a = 9
b = "ali_uzaya_git"
c = a*2

print(c)
print(a/c)
print(a*5)
print(a*c)

print(type(100))
print(type(100.2))
print(type(1+2j))

"""

#Type Dönüşümleri
"""
toplama_bir = input()
toplama_iki = input()

type(toplama_bir)

toplama_bir + toplama_iki

int(toplama_bi.r) + int(toplama_iki)

12

float(12)

str(12)

type(str(12))
"""

# print()
"""
print("geleceği", "yazanlar")
print("geleceği","yazanlar", sep = "_")
"""

#VERİ YAPILARI
"""
#Listeler

#[]
#list()


notlar = [90,80,70,50]

type(notlar)

liste = ["a", 19.3, 90]
l = ["a",19.3,90, notlar]

len(l)

type(l)

type(l[0])

l[0]
l[1]

tum_liste = [liste,l]

#del tum_liste #kod akışı içerisinde silmeyi sağlar.

#Listeler - Eleman İşlemleri

liste = [10,20,30,40,50]

liste[1]
liste[0]

liste[0:2]
liste[:2] #155 ve 156. satırlar aynıdır.

liste[2:] #2.indisten sonrasını alır.

yeni_liste = ["a",10,[20,30,40,50]]
yeni_liste

yeni_liste[3] #3. indis yoktur o liste tektir.

yeni_liste[2]
yeni_liste[0:2]

yeni_liste[2][1] #Listenin içindeki listenin elemanlarına erişmek.


#Listeler - Eleman Değiştirme

liste = ["ali","veli","berkcan","ayse"]
liste

liste[1]
liste[1] = "Velinin_babası"

liste

liste[1] = "Veli"

liste[0:3] = "alinin_babasi", "velinin_babasi", "berkcanin_babasi"
liste

liste = ["ali","veli","berkcan","ayse"]
liste

liste + ["kemal"]
liste

liste = liste + ["kemal"]
liste

#del liste[2]
liste

#Listeler - Liste Metodları

liste = ["ali","veli","isik"]

dir(liste)

liste

#append

liste.append("berkcan")
liste

#remove
liste.remove("berkcan")
liste

#insert

liste = ["ali","veli","isik"]

liste

liste.insert(0,"ayse")
liste

liste.insert(2,"mehmet")
liste

liste.insert(5,"berk")
liste

len(liste)

liste.insert(len(liste),"elif")
liste

#pop

liste.pop(0)
liste

liste.pop(4)
liste

#count

liste = ["ali","veli","isik","ali","veli"]
liste

liste.count("ali")
liste.count("isik")
liste.count("veli")

#copy

liste_yedek = liste.copy()
liste_yedek

#extend

liste.extend(["a","b",10]) #Yeni bir listeyle birleştirir.
liste

#index

liste.index("ali")

#reverse()

liste.reverse() #Terse çevirir
liste

#sort()

liste.sort() #Hem string hem integer olduğu için verdi.
liste = [10,40,5,90]
liste.sort()
liste

#clear

liste.clear()
liste

#Veri Yapıları - Tuple

#Kapsayıcıdır Sıralıdır Değiştirilemez

#Tuple Oluşturma

t = ("ali","veli",1,2,3.2,[1,2,3,4])

t = "ali", "veli", 1,2,3.2,[1,2,3,4]

#tuple()

t = ("eleman",) #Tek eleman olduğunda sonuna , koymak zorundayız.
type(t)

#Tuple Eleman İşlemleri

t = ("ali", "veli", 1,2,3, [1,2,3,4])
t

t[1]
t[0:3]

t[2] = 99 #Değiştirilemez hata verir

#Veri Yapıları - Dictionary(Sözlük)

#Kapsayıcıdır sırasızdır değiştirilebilir.

#Sözlük Oluşturma


sozluk = {"REG":10,
          "LOJ": 20,
          "CART": 30}

sozluk

sozluk = {"REG": ["RMSE:",10],
          "LOJ": ["MSE:",20],
          "CART": ["SSE",30]}

sozluk

#Sozluk Eleman İşlemleri

sozluk = {"REG": "Regresypn Modeli",
          "LOJ": "Lojistik Regresyon",
          "CART": "Classification and Reg"}

sozluk[0] #İndex yapısı olmadığından hata verdi

sozluk["REG"]
sozluk["LOJ"]

sozluk = {"REG": ["RMSE:",10],
          "LOJ": ["MSE:",20],
          "CART": ["SSE",30]}

sozluk["REG"]

sozluk = {"REG": {"RMSE":10,
                  "MSE":20,
                  "SSE":30},

          "LOJ": {"RMSE":10,
                  "MSE":20,
                  "SSE":30},

          "CART": {"RMSE":10,
                  "MSE":20,
                  "SSE":30}}

sozluk

sozluk["REG"]
sozluk["REG"]["SSE"]

#Eleman Ekleme
sozluk = {"REG": "Regresypn Modeli",
          "LOJ": "Lojistik Regresyon",
          "CART": "Classification and Reg"}

sozluk["GBM"] = "Gradient Boosting Machine"
sozluk

#Değiştirme

sozluk["REG"] = "Coklu Dogrusal Regresyon"
sozluk

sozluk[1] = "Yapay Sinir Ağları"
sozluk

l = [1]
l

sozluk[l] = "Yeni Bir şey" #Keyler sabit değer olmak zorunda o yüzden hata verir

t = ("tuple",)

sozluk[t] = "yeni bir şey"
sozluk #Tuple eklenebilir.

#Veri Yapıları - Setler

#Sırasızdır değerleri eşsizdir değiştirilebilirdir farklı tipleri barındırabilir.

#Set oluşturma

s = set()
s

l = [1,"a","ali",123]
s = set(l)
s

t = ("a","ali")

s = set(t)
s

ali = "lutfen_ata_bakma_uzaya_git"
type(ali)

s = set(ali)
s

l = ["ali", "lutfen", "ata", "bakma","uzaya","git","git","ali","git"]
l

s = set(l)
s

len(s)

l[0]

s[0] #sırasızdır index kullanılmaz

#Eleman Eklme ve Çıkarma

l = ["gelecegi", "yazanlar"]

s = set(l)
s

dir(s)

s.add("ile")
s

s.add("gelecege_git")
s

s.add("ile")
s

s.remove("ali")
s

s.remove("ali") #ali yok silmez
s.discard("ali") #silmez bulamadığı için ama hata vermez

#Setler - Klasik Küme İşlemleri

# diffrence() ile iki kümenin farkını ya da "-" ifadesi
# intersection() iki küme kesisimi ya da "&" ifadesi
# union() iki kümenin birleşimi
# symmetric_difference() ikisinde de olmayanları.

#Difference

set1 = set([1,3,5])
set2 = set([1,2,3])

set1.difference(set2)
set2.difference(set1)

#symetric_difference

set1.symmetric_difference(set2)

set1 - set2
set2 - set1

#intersection

set1.intersection(set2)
set2.intersection(set1)

set1 & set2
kesisim = set1 & set2

set1.intersection_update(set2)
set1

#union

birlesim = set1.union(set2)
birlesim

#Setlerde Sorgu İşlemleri

set1 = set([7,8,9])
set2 = set([5,6,7,8,9,10])

#iki kümenin kesişiminin bol olup olmadığının sorgulanması

set1.isdisjoint(set2)

# bir kümenin bütün elemanlarının başka bir küme içerisinde yer alıp almadığı

set1.issubset(set2)

# bir kümenin bir diğer kümeyi kapsayıp kapsamadığı

set2.issuperset(set1)

"""

#FONKSIYONLARA GIRIS VE FONK OKURYAZARLIGI

#Fonksiyon Nasıl Yazılır?

def kare_al(x):
    print(x**2)

kare_al(3)

#Bilgi Notuyla Çıktı Üretmek

def kare_al(x):
    print("Girilen Sayının Karesi:"+str(x**2))

kare_al(5)

def kare_al(x):
    print("Girilen "+str(x)+" Sayının Karesi:"+str(x**2))

kare_al(5)

#İki Argümanlı Fonksiyon Tanimlamak

def kare_al(x):
    print(x**2)

def carpma_yap(x,y):
    print(x*y)

carpma_yap(2,3)

#Ön Tanımlı Argümanlar

def carpma_yap(x,y = 1):
    print(x*y)

carpma_yap(3)

carpma_yap(3,3) #Değer verilirse ön tanımlı yok sayılır.

def carpma_yap(x,y = 1):
    print(x*y)

carpma_yap(y =2,x=3)

#Ne zaman fonksiyon yazılır?

#ısı, nem, şarj

def direk_hesap(isi,nem,sarj):
    print((isi+nem) / sarj)

direk_hesap(25,40,70)


#Çıktıyı Girdi Olarak Kullanmak

def direk_hesap(isi,nem,sarj):
    print((isi+nem) / sarj)

cikti = direk_hesap(25,40,70)
cikti
print(cikti)

def direk_hesap(isi,nem,sarj):
    return (isi+nem) / sarj

direk_hesap(25,40,70)*9

cikti = direk_hesap(25,40,70)
cikti
print(cikti)

#Local ve Global Değişkenler

x = 10
y = 20

def carpma_yap(x,y):
    return x*y

carpma_yap(2,3)

#Local Etki Alanından Global Etki Alanını Değiştirmek

x = []

def eleman_ekle(y):
    x.append(y)
    print(str(y)+ " ifadesi eklendi")

eleman_ekle(1)

eleman_ekle("veli")

x

#Karar & Kontrol Yapıları

#True - False Sorgulamaları

sinir = 5000

sinir == 4000

sinir == 5000

#if
sinir = 5000
gelir = 10000

gelir < sinir

if gelir < sinir:
    print("Gelir sınırdan küçük")

if gelir > sinir:
    print("Gelir sınırdan büyük")

#else
sinir = 50000
gelir = 10000

if gelir > sinir:
    print("Gelir sınırdan büyük")
else:
    print("Gelir sınırdan küçük")

#diger örnek

sinir = 50000
gelir = 50000

if gelir == sinir:
    print("Gelir sinira eşittir")
else:
    print("Gelir sınıra eşit değildir.")

#elif

sinir = 50000
gelir1 = 60000
gelir2 = 50000
gelir3 = 35000

if gelir1 > sinir:
    print("Tebrikler, hediye kazandınız")
elif gelir1 < sinir:
    print("Malesef kazanamadınız")
else:
    print("Gelir sınıra eşittir")

if gelir2 > sinir:
    print("Tebrikler, hediye kazandınız")
elif gelir2 < sinir:
    print("Malesef kazanamadınız")
else:
    print("Gelir sınıra eşittir")

if gelir3 > sinir:
    print("Tebrikler, hediye kazandınız")
elif gelir3 < sinir:
    print("Malesef kazanamadınız")
else:
    print("Gelir sınıra eşittir")

#mini uygulama

sinir = 50000
magaza_adi = input("Mağaza adı nedir?")
gelir = int(input("Geliriniz ne kadar?"))

if gelir > sinir:
    print("Tebrikler "+magaza_adi+ " promosyon kazandınız")
elif gelir < sinir:
    print("Uyarı! Çok düşük gelir:"+ str(gelir))
else:
    print("Çalışmaya devam")

#Döngüler - For

ogrenci = ["ali","veli","isik","berk"]

ogrenci[0]
ogrenci[1]
ogrenci[2]
ogrenci[3]

for i in ogrenci:
    print(i)

#For Örnek

maaslar = [1000,2000,3000,4000,5000]

for maas in maaslar:
    print(maas)

#Döngü ve fonksiyonları birlikte kullanmak

#maaslara %20 zam yapılacak

def yeni_maas(x):
    print(x*20/100+x)

yeni_maas(1000)

for i in maaslar:
    yeni_maas(i)


#mini uygulama
#if, for ve fonksiyonları birlikte kullanmak

maaslar = [1000,2000,3000,4000,5000]

def maas_ust(x):
    print(x*10/100 + x)

def maas_alt(x):
    print(x*20/100 + x)

for i in maaslar:
    if i >= 3000:
        maas_ust(i)
    else:
        maas_alt(i)

#break & continue

maaslar = [8000,5000,2000,3000,1000,3000,7000,1000]

dir(maaslar)

maaslar.sort()
maaslar

for i in maaslar:
    if i == 3000:
        print("kesildi")
        break
    print(i)

for i in maaslar:
    if i == 3000:
        continue
    print(i)

#While

sayi = 1

while sayi < 10:
    sayi += 1
    print(sayi)

# NESNE YÖNELİMLİ PROGRAMLAMA

#Sinif Nedir?

#Sınıf Özellikleri (Class Attributes)

class VeriBilimci():
    bolum = ''
    sql = 'Evet'
    deneyim_yili = 0
    bildigi_diller = []

#Sınıfların özelliklerine erişmek

VeriBilimci.bolum
VeriBilimci.sql

#Sınıfların özelliklerini değiştirmek

VeriBilimci.sql = "Hayır"
VeriBilimci.sql

#Sınıf Örneklendirmesi (instantiation)

ali = VeriBilimci()

ali.sql
ali.deneyim_yili
ali.bolum

ali.bildigi_diller.append("Python")
ali.bildigi_diller

veli = VeriBilimci()
veli.sql

veli.bildigi_diller


#Örnek Özellikleri

class VeriBilimci():
    bildigi_diller = ["R","PYTHON"]
    bolum = ''
    sql = ''
    def __init__(self):
        self.bildigi_diller = []
        self.bolum = ''

ali = VeriBilimci()
ali.bildigi_diller

veli = VeriBilimci()
veli.bildigi_diller

ali.bildigi_diller.append("Python")
ali.bildigi_diller

veli.bildigi_diller
veli.bildigi_diller.append("R")

veli.bildigi_diller
ali.bildigi_diller

VeriBilimci.bildigi_diller


ali.bolum

VeriBilimci.bolum
ali.bolum = 'istatistik'
VeriBilimci.bolum
ali.bolum

veli.bolum
veli.bolum = 'ybs'
veli.bolum
ali.bolum
VeriBilimci.bolum

#Örnek Metodları

class VeriBilimci():
    calisanlar = []
    def __init__(self):
        self.bildigi_diller = []
        self.bolum = ''
    def dil_ekle(self, yeni_dil):
        self.bildigi_diller.append(yeni_dil)

ali = VeriBilimci()
ali.bildigi_diller
ali.bolum

veli = VeriBilimci()
veli.bildigi_diller
veli.bolum

VeriBilimci.dil_ekle("R")

ali.dil_ekle("R")
ali.bildigi_diller

veli.dil_ekle("Python")
veli.bildigi_diller

#Miras Yapıları (inheritance)

class Employees():
    def __init__(self):
        self.FirstName = ""
        self.LastName = ""
        self.Address = ""

class DataScience(Employees):
    def __init__(self):
        self.Programming = ""

veribilimci1 = DataScience()

class Marketing(Employees):
    def __init__(self):
        self.StoryTelling = ""


mar1 = Marketing()


class Employees_yeni():
    def __init__(self,FirstName,LastName,Address):
        self.FirstName = FirstName
        self.LastName = LastName
        self.Address = Address

ali = Employees_yeni("a","b","c")
ali.FirstName

#Python'da Fonksiyonel Programlama
"""
Fonksiyonlar dilin bastacidir.
Birinci sinif nesnelerdir.
Yan etkisiz fonksiyonlar. (stateless, girdi-cıktı)
Yüksek seviye fonksiyonlar
vektörel operasyonlar
"""

#Yan Etkisiz Fonksiyonlar(Pure Functions)

#Ornek1: Yan Etki

A = 9

def impure_sum(b):
    return b + A

def pure_sum(a,b):
    return a + b

impure_sum(5)

pure_sum(3,4)


#Ornek 2: Olumcul
#OOP

class LineCounter:
    def __init__(self,filename):
        self.file = open(filename, 'r')
        self.lines = []

    def read(self):
        self.lines = [line for line in self.file]

    def count(self):
        return len(self.lines)

lc = LineCounter('deneme.txt')

print(lc.lines)
print(lc.count())

lc.read()

print(lc.lines)
print(lc.count())

#FP

def read(filename):
    with open(filename, 'r') as f:
        return [line for line in f]

def count(lines):
    return len(lines)

example_lines = read('deneme.txt')
lines_count = count(example_lines)
lines_count

#İsimsiz Fonksiyonlar (Anonymous Functions)

def old_sum(a,b):
    return a +b

old_sum(4,5)

new_sum = lambda a,b : a+b
new_sum(54,5)

sirasiz_liste = [("b",3),("a",8),("d",12),("c",1)]
sirasiz_liste

sorted(sirasiz_liste, key=lambda x: x[1])

#Vektörel Operasyonlar (Vectorel Operations)
#OOP
a = [1,2,3,4]
b = [2,3,4,5]

ab = []

range(0,len(a))

for i in range(0, len(a)):
    ab.append(a[i]*b[i])

print(ab)

#FP

import numpy as np
a = np.array([1,2,3,4])
b = np.array([2,3,4,5])

ab

#Map & Filter & Reduce

liste = [1,2,3,4,5]

list(map(lambda x: x+10, liste))

#filter

liste = [1,2,3,4,5,6,7,8,9,10]

list(filter(lambda x: x%2 == 0,liste))

#reduce

from functools import reduce
liste = [1,2,3,4]

reduce(lambda a,b: a+b,liste)

#Hatalar / istisnalar(exceptions)

a = 10
b = 0

a/b

try:
    print(a/b)
except ZeroDivisionError:
    print("Payda 0 olmaz")

#Tip hatası
a = 10
b = "2"

a/b

try:
    print(a/b)
except TypeError:
    print("Sayı string bölünmez")
