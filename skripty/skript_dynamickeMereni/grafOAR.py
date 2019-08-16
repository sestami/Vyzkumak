# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:52:22 2019

@author: michal.sestak
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import numpy as np
#import datetime

dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')

soubor='a4'
# data=pd.read_csv(soubor, sep="\t", engine='python', parse_dates=['cas'], date_parser=dateparse)
data=pd.read_csv(soubor+'_modified.csv', date_parser=dateparse, parse_dates=['cas'], index_col='zaznam')

days=mdates.DayLocator()
hours=mdates.HourLocator() #zbytecne, nepouzito
#min(data.iloc[:,1])
#max(data.iloc[:,1])

# fig, ax1 = plt.subplots(figsize=(25,10))
fig, ax1 = plt.subplots(figsize=(14,7))
#ax1.xaxis_date()
#def make_patch_spines_invisible(ax):
 #   ax.set_frame_on(True)
  #  ax.patch.set_visible(False)
  #  for sp in ax.spines.values():
  #      sp.set_visible(False)

fig.subplots_adjust(right=0.75)

ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.08))

#make_patch_spines_invisible(par2)
ax3.spines["right"].set_visible(True)

p1,=ax1.plot(data.loc[:,'cas'],data.loc[:,'radon[Bq/m3]'],'r',linewidth=2)
p2,=ax2.plot(data.loc[:,'cas'],data.loc[:,'Teplota[C]'],'g',linewidth=0.8)
p3,=ax3.plot(data.loc[:,'cas'],data.loc[:,'Vlhkost[%]'],'b',linewidth=0.8)

ax1.xaxis.set_major_locator(days)
#ax1.xaxis.set_minor_locator(hours)

ax1.set_yticks(np.linspace(0, ax1.get_ybound()[1], 10))
ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 10))
ax3.set_yticks(np.linspace(ax3.get_ybound()[0], ax3.get_ybound()[1], 10))

#rounding tick's labels
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#ax2.set_yticks(np.linspace(par1.get_yticks()[0], par1.get_yticks()[-1], len(host.get_yticks())))
#ax3.set_yticks(np.linspace(par2.get_yticks()[0], par2.get_yticks()[-1], len(host.get_yticks())))


ax1.set_xlabel("$datum$")
ax1.set_ylabel("$OAR$ [Bq/m$^3$]")
ax2.set_ylabel(r"$T$ [$^{\circ}$C]")
ax3.set_ylabel("$rH$ [%]")

ax1.yaxis.label.set_color(p1.get_color())
ax2.yaxis.label.set_color(p2.get_color())
ax3.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
ax1.tick_params(axis='y', colors=p1.get_color(), **tkw)
ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax1.tick_params(axis='x', **tkw)

lines = [p1, p2, p3]

# ax1.legend(lines, [l.get_label() for l in lines])

format_data = mdates.DateFormatter('%d. %m. %H:%M')
ax1.xaxis.set_major_formatter(format_data)
fig.autofmt_xdate()

ax1.grid()
plt.savefig(soubor+'.png',format='png',dpi=200, bbox_inches='tight')
# ax1.legend_.remove()
