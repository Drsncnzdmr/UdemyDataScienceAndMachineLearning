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

gel_yaz = "geleceği yazanlar"

gel_yaz_cap = gel_yaz.capitalize() #Baş harfi büyütür.
print(gel_yaz_cap)

gel_yaz_title = gel_yaz.title() #Her başlangıçı büyüttü.
print(gel_yaz_title)
