import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pylab import * 
import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

plt.rcParams["font.family"] = "Times New Roman"

# Import of the data
filepath = 'data/WHR.csv'
data = pd.read_csv(filepath, sep = ',')

# info of the data
data.head()
data.shape
data.info()

# Skewed data?
float_cols = data.dtypes[data.dtypes == np.float]
float_cols_list = float_cols.index.tolist()
skew_limit = 0.75
data_skw = data[float_cols_list].skew()

# As we did in the class exercicise, we will skew the variables
skew_cols = (data_skw
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > 0.75'))
skew_cols

# Now we apply the log to these variables

skew_cols.index.values 

dataS = data.copy()
for col in skew_cols.index.values:
    dataS[col] = data[col].apply(np.log1p)
    
    
fig, axs = plt.subplots(1, 3, figsize = (20, 7))    

dataS['Generosity'].hist( ax = axs[0], bins= 25, alpha = 0.5, color = 'blue',density=True, edgecolor='white')
dataS['Social support'].hist( ax = axs[1], bins= 25, alpha = 0.5, color = 'blue',density=True, edgecolor='white')
dataS['Perceptions of corruption'].hist( ax = axs[2], bins= 25, alpha = 0.5, color = 'blue',density=True, edgecolor='white')

 
for i, ax in enumerate(axs.flat):
    ax.set_xlabel('Year').set_fontsize(28)
    ax.set_ylabel('Frequency').set_fontsize(28)

axs[0].set_title('Generosity_after_skew', pad=15).set_fontsize(28)
axs[1].set_title('Social support_after_skew', pad=15).set_fontsize(28)
axs[2].set_title('Perceptions of corruption_after_skew', pad=15).set_fontsize(28)

fig.tight_layout(pad=2.5)

plt.savefig('skew1.png', bbox_inches='tight')

fig, axs = plt.subplots(1, 3, figsize = (20, 7))    

data['Generosity'].hist( ax = axs[0], bins= 25, alpha = 0.5, color = 'blue' ,density=True, edgecolor='white')
data['Social support'].hist( ax = axs[1], bins= 25, alpha = 0.5, color = 'blue',density=True, edgecolor='white')
data['Perceptions of corruption'].hist( ax = axs[2], bins= 25, alpha = 0.5, color = 'blue',density=True, edgecolor='white')

 
for i, ax in enumerate(axs.flat):
    ax.set_xlabel('Year').set_fontsize(28)
    ax.set_ylabel('Frequency').set_fontsize(28)

axs[0].set_title('Generosity_before_skew', pad=15).set_fontsize(28)
axs[1].set_title('Social support_before_skew', pad=15).set_fontsize(28)
axs[2].set_title('Perceptions of corruption_before_skew', pad=15).set_fontsize(28)

fig.tight_layout(pad=2.5)

plt.savefig('skew2.png', bbox_inches='tight')

dataS['Perceptions of corruption'].skew()

# Now pairplot

df = data.copy()
df['Country Seg']=df['Country name']
df.loc[~df['Country Seg'].isin(['Burundi','Luxembourg', 'Mexico', 'Malawi', 'Australia', 'Bulgaria']),'Country Seg']='Other'

# PAIRPLOT SEGREGATED
fig.set_size_inches(70,35)
sns.set_context("paper", rc={"axes.labelsize":15})
ax = sns.pairplot(
    df.sort_values('Country Seg', ascending=False),
    hue='Country Seg'
    )

fig.legend(['Burundi','Luxembourg', 'Mexico', 'Malawi', 'Australia', 'Bulgaria'], bbox_to_anchor=(0., 1.0, 0, .102), loc='lower left',
           ncol=5, fontsize=35)
plt.savefig('PairPlot.png', bbox_inches='tight')

# Evolutions in time for all countries
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,5))

data.groupby('year').mean().plot(y='Life Ladder', ax=ax1, color='blue', style='.-',  markersize=20)
ax1.axis([2006, 2021, 5.2, 6.0])
ax1.set_xlabel('Year').set_fontsize(25)

data.groupby('year').mean().plot(y='Log GDP per capita', ax=ax2, color='black', style='.-',  markersize=20)
ax2.axis([2006, 2021, 9.0, 10.0])
ax2.set_xlabel('Year').set_fontsize(25)

data.groupby('year').mean().plot(y='Healthy life expectancy at birth', ax=ax3, color='red', style='.-',  markersize=20)
ax3.axis([2006, 2021, 60, 68])
ax3.set_xlabel('Year').set_fontsize(25)

ax1.legend(loc=2, prop={'size': 18})
ax2.legend(loc=2, prop={'size': 18})
ax3.legend(loc=2, prop={'size': 15})

fig.tight_layout(pad=2.5)

plt.savefig('year.png', bbox_inches='tight')

# hyppothesis
import geopandas as gpd

countriesDF=gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
countriesDF['latitude']=countriesDF.geometry.centroid.y
countriesDF['longitude']=countriesDF.geometry.centroid.x

# We change some names from the original data to fit the names of he countries data frame
# from geopandas. There are still some countries left
data.loc[data['Country name']=='United States','Country name']='United States of America'
data.loc[data['Country name']=='Congo (Kinshasa)', 'Country name']='Dem. Rep. Congo'

# in Here we merge both data frames. there will be some countries without latitude and longitude
# data.
dataextraDF=(gpd.GeoDataFrame(
     data
    .merge(
        countriesDF[['name','latitude', 'continent','pop_est','geometry']],
        left_on='Country name',
        right_on='name',
        how='inner')#left
    .drop(columns='name')
    ))
    
# #### Hypothesis 1: Does Life Ladder depend on latitude?

dataextra = dataextraDF.copy()
dataextra.loc[dataextra['latitude']<0, 'latitude'] = abs(dataextra['latitude'])
dataextra.latitude.min()

ax = plt.figure(figsize=(10, 5))
plt.plot(dataextra['latitude'], dataextra['Life Ladder'], 'o', color = 'yellow', 
         markeredgecolor='black', markersize=10)

sns.regplot(x="latitude", y="Life Ladder", data=dataextra, color = 'blue')


plt.xlabel('Latitude', fontsize=25)
plt.ylabel('Life Ladder', fontsize=25)
plt.axis([-5, 70, 2, 9])

plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 

plt.savefig('Ladderlatitude.png', bbox_inches='tight')

X2 = sm.add_constant(dataextra['latitude'])
est = sm.OLS(dataextra['Life Ladder'], X2)
est2 = est.fit()
print(est2.summary())

dataextra.plot('Life Ladder', legend=True, legend_kwds={'label': "Life Ladder", 'orientation': "horizontal"}, figsize=(20,10))
plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 
plt.savefig('LifeLadder.png', bbox_inches='tight')


##### #### Hypothesis 2: Does Social support depend on GDP?

dataextra2 = dataextraDF.copy()
dataextra2.dropna(axis=1)
dataextra2=dataextra2.loc[(dataextra2['Social support'].notnull()) & (dataextra2['Log GDP per capita'].notnull())]
dataextra2.info()

ax = plt.figure(figsize=(10, 5))
plt.plot(dataextra2['Social support'], dataextra2['Log GDP per capita'], 'o', color = 'yellow', 
         markeredgecolor='black', markersize=10)


sns.regplot(x="Social support", y="Log GDP per capita", data=dataextra2, color = 'blue')


plt.xlabel('Social support', fontsize=25)
plt.ylabel('Log GDP per capita', fontsize=25)
plt.axis([0.2, 1.1, 5, 13])

plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 


plt.savefig('GDPsocial.png', bbox_inches='tight')

X2 = sm.add_constant(dataextra2['Social support'])
est = sm.OLS(dataextra2['Log GDP per capita'], X2)
est2 = est.fit()
print(est2.summary())

## #### Hypothesis 2: Does Healthy Life depend on latitude?

dataextra3 = dataextraDF.copy()
#dataextra3 = dataextra3.loc[dataextra3['latitude']>=0]
dataextra3.loc[dataextra3['latitude']<0, 'latitude'] = abs(dataextra3['latitude'])

dataextra3=dataextra3.loc[(dataextra3['latitude'].notnull()) & (dataextra3['Healthy life expectancy at birth'].notnull())]


# only latitudes above 0

dataextra3.info()

ax = plt.figure(figsize=(10, 5))
plt.plot(dataextra3['latitude'], dataextra3['Healthy life expectancy at birth'], 'o', color = 'yellow', 
         markeredgecolor='black', markersize=10)


sns.regplot(x="latitude", y="Healthy life expectancy at birth", data=dataextra3, color = 'blue')


plt.xlabel('latitude', fontsize=25)
plt.ylabel('Healthy life expectancy at birth', fontsize=25)
#plt.axis([-50, 80, 0, 1.05])

plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 


plt.savefig('LifeexpLatitude.png', bbox_inches='tight')

X2 = sm.add_constant(dataextra3['latitude'])
est = sm.OLS(dataextra3['Healthy life expectancy at birth'], X2)
est2 = est.fit()
print(est2.summary())

dataextra.plot('Healthy life expectancy at birth', legend=False, figsize=(20,20))
plt.savefig('LifeMap.png', bbox_inches='tight')
