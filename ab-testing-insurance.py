##AB TEST INSURANCE


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


##VERİ SETİ HİKAYESİ:

#Farklı yaştaki,cinsiyetteki hastaların tedavi maliyetlerine ayrılmış bir veri setini inceleyeceğiz.
#Hastaların teşhisine ilişkin verilerimiz yok.
# Ancak AB testini uygulayabileceğimiz başka bilgilere sahibiz.

#VERİ SETİNDEKİ DEĞİŞKENLER;

#age : Hastaların yaşı
# sex : Hastaların cinsiyeti
# bmi(body mass index): Vücut Kitle Endeksi
# children :Hastaların çocuk sayısı
# smoker : Sigara içme durumu (Yes or No)
# region : Bulundukları bölge
# charges : Hastane masrafı



###Veriyi Anlama ve Hazırlama

df_= pd.read_csv(r"C:\Users\elifd\PycharmProjects\pythonProject1\examples\insurance.csv")
df =df_.copy()
df.head()
df.tail()
df.describe().T
df.shape
df.isnull().sum()


df.groupby("sex").agg({"charges": "mean"})
#matematiksel olarak bir fark görüyorz.

sns.boxplot(x=df["sex"], y=df["charges"])
plt.show()

#AB Testing (Bağımsız İki Örneklem T Testi)
#1-Hipotez Kur

# H0: M1 = M2 (Cinsiyetlerin Hastane Masrafları Arasında İst Ol An Fark yoktur.)
# H1: M1 != M2 (.... vardır)

#2- Varsayım Kontrolü
# Normallik Varsayımı   # shapiro testi

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "charges"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #p value 0.00 o yüzden Ho RED,Normal dağılım sağlanmamaktadır.


test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "charges"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #p value 0.00 o yüzden Ho RED,Normal dağılım sağlanmamaktadır.

## Varyans Homojenligi Varsayımı

# H0: Varyanslar Homojendir.
# H1: ... değildir.

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "charges"],
                           df.loc[df["sex"] == "male", "charges"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#3-Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "charges"],
                                 df.loc[df["sex"] == "male", "charges"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#HO REDDEDİLEMEZ.OLASI FARKLILIK ŞANS ESERİ ORTAYA ÇIKMIŞTIR.
