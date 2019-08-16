import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import datetime
from uncertainties import ufloat
import matplotlib as mpl
mpl.style.use('classic')

dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')

data1=pd.read_csv('a1_modified.csv', date_parser=dateparse,
                 parse_dates=['cas'], index_col='zaznam', comment='#')
data2=pd.read_csv('a2_modified.csv', date_parser=dateparse,
                 parse_dates=['cas'], index_col='zaznam', comment='#')
data3=pd.read_csv('a3_modified.csv', date_parser=dateparse,
                 parse_dates=['cas'], index_col='zaznam', comment='#')
data4=pd.read_csv('a4_modified.csv', date_parser=dateparse,
                 parse_dates=['cas'], index_col='zaznam', comment='#')

fig, ax = plt.subplots(figsize=(13,7))
def plotovani(data,ID):
    ax.plot(data.loc[:,'cas'],data.loc[:,'radon[Bq/m3]'],label=ID)
    data=data.loc[:,'radon[Bq/m3]']
    data.name=ID
    return data

OAR1=plotovani(data1, 'sklep')
OAR2=plotovani(data2, 'přízemí kuchyň')
OAR3=plotovani(data3, 'přízemí ložnice')
OAR4=plotovani(data4, 'první patro')

ax.set_xlabel("$datum$")
ax.set_ylabel("$OAR$ [Bq/m$^3$]")
ax.legend(loc='best')
ax.grid()
# ax.grid(b=True, which='major', linestyle='--')
# ax.grid(b=True, which='minor', linestyle=':')
# plt.minorticks_on()

format_data = mdates.DateFormatter('%d. %m. %H:%M')
ax.xaxis.set_major_formatter(format_data)
fig.autofmt_xdate()

plt.savefig('OAR_dohromady.png', format='png', dpi=200,
            bbox_inches='tight')
plt.show()

# stat1=OAR1.describe()
# stat2=OAR2.describe()
# stat3=OAR3.describe()
# stat4=OAR4.describe()

# statistiky=pd.DataFrame([stat8, stat10, stat88, stat112, statAG])
#statistiky.to_latex('tab.tex',float_format='%0.0f')

def unifikace(stat):
    return ufloat(stat.loc['mean'], stat.loc['std'])

# ID=[8, 10, 88, 112]
# mean_AG=unifikace(statAG)
# mean_sondy=np.array([unifikace(stat8), unifikace(stat10), unifikace(stat88), unifikace(stat112)])
# mean_sondy=pd.DataFrame(mean_sondy,index=ID,columns=['mean'])
# B=mean_AG/mean_sondy # bulharska konstanta
# n=[el.n for el in B['mean'].values]
# s=[el.s for el in B['mean'].values]
# B=pd.DataFrame(np.array([n,s]).T, index=ID, columns=['B', 'B_err'])
# #B.to_latex('tab_bulharska_konst.tex',float_format='%0.3f')


# fig2, ax2 = plt.subplots()
# index=[str(el) for el in B.index]
# #ax2.errorbar(index, B['B'], yerr=B['B_err'], fmt='o')
# ax2.plot(index, B['B'],'rx')
# ax2.axhline(y=1, color='black')
# ax2.grid(axis='y')
# ax2.set_xlabel("ID sondy")
# ax2.set_ylabel("$B$ [-]")
