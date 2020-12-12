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


sozluk

len(sozluk)

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

