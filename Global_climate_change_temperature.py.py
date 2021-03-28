#Importing Libraries
import pandas as pd 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns 

#Scanning Report of Temperature and Seggregating them properly
df = pd.read_csv('GlobalLandTemperatures_GlobalLandTemperaturesByState.csv')
df.head()
df.dtypes
df.shape
df.isnull().sum()
df = df.dropna(how='any',axis=0)
df.shape
df.rename(columns={'dt':'Date','AverageTemperature':'Avg_Temp','AverageTemperatureUncertainity':'confidence_interval_temp'},inplace=True)
df.head()

#Graph after seggregation
df['Date']=pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index
df.describe()
df['Year'] = df.index.year
df.head()
df.describe()
latest_df = df.loc['1900':'2013']
latest_df.head()
latest_df[['Country','Avg_Temp']].groupby(['Country']).mean().sort_values('Avg_Temp')
plt.figure(figsize=(9,4))
sns.lineplot(x="Year",y="Avg_Temp",data=latest_df)
plt.show()
resample_df = latest_df[['Avg_Temp']].resample('A').mean()
resample_df.head()
resample_df.plot(title='Temperature Changes from 1980-2013',figsize=(8,5)) 
plt.ylabel('Temperature',fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.legend()

#Dickey Fuller Test

from statsmodels.tsa.stattools import adfuller
print("Dickey Fuller Test Results: ")
test_df = adfuller(resample_df.iloc[:,0].values, autolag='AIC')
df_output = pd.Series(test_df[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
for key, value in test_df[4].items():
    df_output['Critical Value (%s)'%key]=value
print(df_output)
decomp = seasonal_decompose(resample_df,freq=3)

trend = decomp.trend
seasonal = decomp.seasonal
residual = decomp.resid

plt.subplot(411)
plt.plot(resample_df)
plt.xlabel('Original')
plt.figure(figsize=(6,5))

plt.subplot(412)
plt.plot(trend)
plt.xlabel('Trend')
plt.figure(figsize=(6,5))

plt.subplot(412)
plt.plot(seasonal)
plt.xlabel('Seasonal')
plt.figure(figsize=(6,5))

plt.subplot(414)
plt.plot(residual)
plt.xlabel('Residual')
plt.figure(figsize=(6,5))

plt.tight_layout()

rol_mean = resample_df.rolling(window=3, center=True).mean()
ewm = resample_df.ewm(span=3).mean()

rol_std = resample_df.rolling(window=3, center=True).std()

#Graphical Representation

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

ax1.plot(resample_df, label='Original')
ax1.plot(rol_mean,label='Rolling Mean')
ax1.plot(ewm, label='Exponential Weighted Mean')
ax1.set_title('Temperature Changes from 1980-2013',fontsize=14)
ax1.set_ylabel('Temperature',fontsize=12)
ax1.set_xlabel('Year',fontsize=12)
ax1.legend()

ax2.plot(rol_std,label='Rolling STD')
ax2.set_title('Temperature Changes from 1980-2013',fontsize=14)
ax2.set_ylabel('Temperature',fontsize=12)
ax2.set_xlabel('Year',fontsize=12)
ax2.legend()

plt.tight_layout()
plt.show()

rol_mean.dropna(inplace=True)
ewm.dropna(inplace=True)

#Dickey Fuller Test
print(" Dickey Fuller Test for the Rolling Mean:")
df_test =adfuller(rol_mean.iloc[:,0].values,autolag='AIC')
df_output = pd.Series(df_test[0:4], index=['Test Statistics','p-value','Lags Used','Number of Observations Used'])
for key, value in df_test[4].items():
    df_output['Critical Value (%s)'%key] = value
print(df_output)
print('')
print('Dickey Fuller Test for the Exponential Weighted Mean:')
df_test = adfuller(ewm.iloc[:,0].values, autolag='AIC')
df_output = pd.Series(df_test[0:4], index=['Test Statistics','p-value','Lags Used','Number of Observations Used'])
for key, value in df_test[4].items():
    df_output['Critical Value (%s)'%key] = value
print(df_output)

diff_rol_mean = resample_df - rol_mean
diff_rol_mean.dropna(inplace = True)
diff_rol_mean.head()

diff_ewm = resample_df -ewm
diff_ewm.dropna(inplace=True)
diff_ewm.head()

df_rol_mean_diff = diff_rol_mean.rolling(window=3, center=True).mean()

df_ewm_diff = diff_ewm.ewm(span=3).mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

ax1.plot(diff_rol_mean, label='Original')
ax1.plot( df_rol_mean_diff,label='Rolling Mean')
ax1.set_title('Temperature Changes from 1980-2013',fontsize=14)
ax1.set_ylabel('Temperature',fontsize=12)
ax1.set_xlabel('Year',fontsize=12)
ax1.legend()

ax2.plot(diff_ewm,label='Original')
ax2.plot(df_ewm_diff,label='Exponentially Weighted Mean')
ax2.set_title('Temperature Changes from 1980-2013',fontsize=14)
ax2.set_ylabel('Temperature',fontsize=12)
ax2.set_xlabel('Year',fontsize=12)
ax2.legend()

plt.tight_layout()
plt.show()

#Dickey Fuller Test
print(" Dickey Fuller Test for the difference between the Original and Rolling Mean:")
dftest =adfuller(diff_rol_mean.iloc[:,0].values,autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistics','p-value','Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = values
print(dfoutput)
print('')
print('Dickey Fuller Test for the Original and Exponential Weighted Mean:')
dftest = adfuller(diff_ewm.iloc[:,0].values, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistics','p-value','Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot

pyplot.figure(fidsize=(10,5))
pyplot.subplot(211)
plot_acf(resample_df, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(resample_df, ax= pyplot.gca())
pyplot.show()