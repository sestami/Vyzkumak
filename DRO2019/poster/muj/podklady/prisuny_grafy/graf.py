import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from uncertainties import ufloat

# data=pd.read_csv('Q_skala75.csv')
# data=pd.read_csv('Q_skala75_CANARY.csv')
# data=pd.read_csv('Q_halkova980.csv')
# data=pd.read_csv('Q_halkova980_CANARY.csv')
# data=pd.read_csv('Q_anglicka574.csv')
data=pd.read_csv('Q_anglicka574_CANARY.csv')
for column in data.columns[1:]:
    data.loc[:, column]=[ufloat(x) for x in data.loc[:, column]]

def nominal_value(series):
    return [x.n for x in series]

def std(series):
    return [x.s for x in series]

# zdroje=[ufloat(400, 51), ufloat(114, 13), ufloat(0, 0)]
# zdroje=[ufloat(332, 64), ufloat(0, 0), ufloat(0, 0), ufloat(0, 0)]
zdroje=[ufloat(455, 90), ufloat(0, 0), ufloat(0, 0)]
pocet_kombinaci=1
N=3
zdroje=np.reshape(zdroje*pocet_kombinaci, (pocet_kombinaci, N))

#graf
# fig, ax = plt.subplots(figsize=(7,8))
fig, ax = plt.subplots()
# colors = cm.rainbow(np.linspace(0, 1, pocet_kombinaci))
colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']#, 'tab:olive', 'tab:cyan']
markers=np.concatenate((['x']*4,['+']*4))
# markers=np.concatenate((['x'],['+']))
for i, c in enumerate(colors[:pocet_kombinaci]):
    # for j in np.arange(3):
    # ax.scatter(nominal_value(zdroje[i,:]), nominal_value(data.iloc[i, 1:]), color=c, label=data.iloc[i, 0])
    ax.errorbar(nominal_value(zdroje[i,:]), nominal_value(data.iloc[i, 1:]), yerr=std(data.iloc[i, 1:]), label=data.iloc[i, 0],
                capsize=5, elinewidth=1, markeredgewidth=2, fmt=markers[i],
                ms=10, color=c)
# plt.scatter(nominal_value(zdroje[:,0]), nominal_value(data.loc[:, 'Q_0']), color=c)
ax.plot([-10, 500], [-10, 500], color='black', label='$y=x$')
plt.grid()
plt.legend()
plt.xlabel('Známé přísuny radonu [Bq/m$^3$/hod]')
plt.ylabel('Určené přísuny radonu [Bq/m$^3$/hod]')
plt.tight_layout()

#humpolec chata
# ax.text(380, -20, 'sklep', fontsize=15, bbox={'alpha': 0.5, 'pad': 5})
# ax.text(100, -20, 'přízemí', fontsize=15, bbox={'alpha': 0.5, 'pad': 5})
# ax.text(-20, -20, 'první patro', fontsize=15, bbox={'alpha': 0.5, 'pad': 5})

#humpolec byt
# ax.text(270, -600, 'obývací pokoj', fontsize=15, bbox={'alpha': 0.5, 'pad': 5})

#dobrichovice
ax.text(430, -20, 'sklep', fontsize=15, bbox={'alpha': 0.5, 'pad': 5})

ax.ticklabel_format(useOffset=False)
plt.show()
