# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:03:04 2019

@author: michal.sestak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import datetime
from uncertainties import ufloat

dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')

data8 = pd.read_csv('RDN_17008.tab', encoding="ISO-8859-1",
                    sep="\t",index_col='záznam', parse_dates=['cas'], date_parser=dateparse)
data10 = pd.read_csv('RDN_17010.tab', encoding="ISO-8859-1",
                    sep="\t",index_col='záznam', parse_dates=['cas'], date_parser=dateparse)
data88 = pd.read_csv('RDN_17088.tab', encoding="ISO-8859-1",
                    sep="\t",index_col='záznam', parse_dates=['cas'], date_parser=dateparse)
data112 = pd.read_csv('RDN_17112.tab', encoding="ISO-8859-1",
                    sep="\t",index_col='záznam', parse_dates=['cas'], date_parser=dateparse)

AG = pd.read_csv('alphaguard.csv', sep=';', parse_dates=['cas'], date_parser=dateparse, encoding="ISO-8859-1", dtype={'radon[Bq/m3]': float})
AG.index += 1

fig, ax1 = plt.subplots()
def plotovani(data,ID):
    ax1.plot(data.loc[3:73,'cas'],data.loc[3:73,'radon[Bq/m3]'],label=ID)
    data=data.loc[3:73,'radon[Bq/m3]']
    data.name=ID
    return data

OAR8=plotovani(data8, '8')
OAR10=plotovani(data10, '10')
OAR88=plotovani(data88, '88')
OAR112=plotovani(data112, '112')
OAR_AG=plotovani(AG, 'AlphaGuard')

ax1.set_xlabel("$datum$")
ax1.set_ylabel("$OAR$ [Bq/m$^3$]")
ax1.legend()

stat8=OAR8.describe()
stat10=OAR10.describe()
stat88=OAR88.describe()
stat112=OAR112.describe()
statAG=OAR_AG.describe()

statistiky=pd.DataFrame([stat8, stat10, stat88, stat112, statAG])
#statistiky.to_latex('tab.tex',float_format='%0.0f')

def unifikace(stat):
    return ufloat(stat.loc['mean'], stat.loc['std'])

ID=[8, 10, 88, 112]
mean_AG=unifikace(statAG)
mean_sondy=np.array([unifikace(stat8), unifikace(stat10), unifikace(stat88), unifikace(stat112)])
mean_sondy=pd.DataFrame(mean_sondy,index=ID,columns=['mean'])
B=mean_AG/mean_sondy # bulharska konstanta
n=[el.n for el in B['mean'].values]
s=[el.s for el in B['mean'].values]
B=pd.DataFrame(np.array([n,s]).T, index=ID, columns=['B', 'B_err'])
#B.to_latex('tab_bulharska_konst.tex',float_format='%0.3f')

format_data = mdates.DateFormatter('%d. %m. %Y, %H:%M')
ax1.xaxis.set_major_formatter(format_data)
fig.autofmt_xdate()
ax1.grid()

fig2, ax2 = plt.subplots()
index=[str(el) for el in B.index]
#ax2.errorbar(index, B['B'], yerr=B['B_err'], fmt='o')
ax2.plot(index, B['B'],'rx')
ax2.axhline(y=1, color='black')
ax2.grid(axis='y')
ax2.set_xlabel("ID sondy")
ax2.set_ylabel("$B$ [-]")
