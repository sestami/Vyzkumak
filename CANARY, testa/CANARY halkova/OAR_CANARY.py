import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import datetime
from uncertainties import ufloat
import matplotlib as mpl
mpl.style.use('classic')
#mpl.style.use('default')
import datetime

dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')


data1=pd.read_csv('Akce HuSk_ No.1_2019.csv', comment='#',
                  skiprows=10, sep=';', decimal=',')
data2=pd.read_csv('Akce HuSk_ No.2_2019.csv', comment='#',
                  skiprows=10, sep=';', decimal=',')
data3=pd.read_csv('Akce HuSk_ No.3_2019.csv', comment='#',
                  skiprows=10, sep=';', decimal=',')
data4=pd.read_csv('Akce HuSk_ No.4_2019.csv', comment='#',
                  skiprows=10, sep=';', decimal=',')


rozdily=np.array([datetime.timedelta(hours=el) for el in data1.loc[:, 'TIME']])
pocatek=datetime.datetime(2019,5,23,12,0,0)
datumy=pocatek+rozdily

data1.index=datumy
data1.loc[:, 'TIMESTAMP']=datumy
data2.index=datumy
data2.loc[:, 'TIMESTAMP']=datumy
data3.index=datumy
data3.loc[:, 'TIMESTAMP']=datumy
data4.index=datumy
data4.loc[:, 'TIMESTAMP']=datumy

poc='2019-06-05 13:00'
kon='2019-06-20 10:00'
fig, ax = plt.subplots(figsize=(12,7))
def plotovani(data,ID):
    ax.plot(data.loc[poc:kon, 'TIMESTAMP'], data.loc[poc:kon, 'RADON CONCENTRATION'],label=ID)
    data=data.loc[poc:kon,'RADON CONCENTRATION']
    data.name=ID
    return data

print('Nezapomen zmenit legendu!')
OAR1=plotovani(data1, 'obývací pokoj')
OAR2=plotovani(data2, 'ložnice')
OAR3=plotovani(data3, 'koupelna+WC')
OAR4=plotovani(data4, 'kuchyň')

ax.set_xlabel("$datum$")
ax.set_ylabel("$OAR$ [Bq/m$^3$]")
ax.legend(loc='best')
ax.grid()
ax.set_title('CANARY měřáky')
ax.set_ylim(bottom=-50)
#ax.grid(b=True, which='major', linestyle='--')
#ax.grid(b=True, which='minor', linestyle=':')
## plt.minorticks_on()
#
format_data = mdates.DateFormatter('%d. %m. %H:%M')
ax.xaxis.set_major_formatter(format_data)
fig.autofmt_xdate()
#
plt.savefig('OAR_CANARY.png', format='png', dpi=200,
          bbox_inches='tight')
plt.show()

OAR_means=np.array([OAR1.mean(), OAR2.mean(), OAR3.mean(), OAR4.mean()])
OAR_sigma=0.1*OAR_means
#
#T1=data1.loc[:, 'Teplota[C]']
#T2=data2.loc[:, 'Teplota[C]']
#T3=data3.loc[:, 'Teplota[C]']
#
#T1_mean=T1.mean()
#T2_mean=T2.mean()
#T3_mean=T3.mean()
