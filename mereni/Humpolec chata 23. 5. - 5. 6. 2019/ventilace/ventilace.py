import numpy as np
import pandas as pd
import datetime
from scipy.optimize import lsq_linear
from uncertainties import ufloat, unumpy
import statsmodels.api as sm

'''
VYPOCET PRUTOKU VZDUCHU POMOCI METODY NEJMENSICH CTVERCU
'''

def completion(velicina, velicina_err):
    united = np.array([ufloat(value, error) for value, error in zip(velicina.to_numpy(), velicina_err.to_numpy())])
    return pd.Series(data=united, index=velicina.index, name=velicina.name)

#ZADANI VSTUPNICH PARAMETRU, VYPOCET KONCENTRACI
odezva_TD = pd.read_csv('odezva_TD_detektoru.csv',sep=';', decimal=',',index_col=['zona', 'plyn'])
# odezva_TD = pd.read_csv('odezva_TD_detektoru_nepreurcena.csv',sep=';', decimal=',',index_col=['zona', 'plyn'])
odezva_TD = odezva_TD.swaplevel().sort_index()
informace_plyny = pd.read_csv('plyny_informace.csv',sep=';', decimal=',',index_col='plyn')
# informace_plyny = pd.read_csv('plyny_informace_nepreurcena.csv',sep=';', decimal=',',index_col='plyn')

N=len(odezva_TD.index.levels[1]) #pocet zon
N_plyny=len(odezva_TD.index.levels[0]) #pocet pouzitych indikacnich plynu

#vypocet doby mereni
pocatek=datetime.datetime(2019, 5, 23, 12, 00)
konec=datetime.datetime(2019, 6, 5, 10, 45)
rozdil=konec-pocatek
dT=rozdil.total_seconds()/3600 #v hod

'''
Vsechno jsou to vektory o velikosti N x (pocet pouzitych indikacnich plynu),
    krome dT, coz je skalar
[R_d]=ng
[dT]=min
[M_w]=g/mol
[T]=K
[p]=Pa
[odpary]=mg, vektor
[dT]=hod, skalar
'''
#velicina   rozmer
R_d    =odezva_TD['R_d'] #N_plyny x N
R_d_err=odezva_TD['R_d_err'] #N_plyny x N
p      =odezva_TD['p']   #N_plyny x N
T      =odezva_TD['T']+273.15 #N_plyny x N
odpary =odezva_TD['odpary'] #N_plyny x N

R_d=completion(R_d, R_d_err)

U_R   =informace_plyny['U_R'] #N_plyny
M_w   =informace_plyny['M_w'] #N_plyny
#vse to proste musi mit rozmer stejny jako C

R_konst=8.314462618 #univerzalni plynova konstanta [J/K/mol]

#koncentrace a emise
C=np.full((N_plyny, N), np.nan, dtype=object)
M=np.full(N*N_plyny, np.nan, dtype=object)
for k in np.arange(1, N_plyny+1):
    for i in np.arange(1,N+1):
        C[k-1,i-1]=R_d[k,i]/(U_R[k]*dT*60)*M_w[k]*p[k,i]/(R_konst*1000*T[k,i])
C=pd.DataFrame(C, index=R_d.index.levels[0], columns=R_d.index.levels[1])
M=odpary/dT

#veliciny kolem prutoku vzduchu
pocet_neznamych=N*N

index1,index2=[],[]
for i in np.arange(1,N+1):
    for j in np.arange(1,N+2):
        if i!=j:
            index1.append(i)
            index2.append(j)

#indexovani prutoku a jejich poloha v vektoru reseni, tj. v podstate takova funkce, mapovani
prutoky_indexy=pd.DataFrame(np.array([index1, index2]).T, index=np.arange(1,pocet_neznamych+1),
                            columns=['vychozi zona', 'cilova zona'])

#matice Z
Z_n=pd.DataFrame(np.zeros((N*N_plyny, pocet_neznamych)), index=odezva_TD.index)
Z_n.columns+=1
Z_s=Z_n.copy()

for m in np.arange(1, pocet_neznamych+1):
    i, j=prutoky_indexy.loc[m]
    for k in np.arange(1,N_plyny+1):
        if i!=j:
            Z_n.loc[(k, i), m]=-C.loc[k, i].n
            Z_s.loc[(k, i), m]=-C.loc[k, i].s
            if j<=N:
                Z_n.loc[(k, j), m]= C.loc[k, i].n
                Z_s.loc[(k, j), m]= C.loc[k, i].s

#nalezeni prutoku metodou nejmensich ctvercu
X=Z_n
y=-M

#rucne
prutoky1=np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

#scipy lsq_linear
prutoky2=lsq_linear(X, y)
prutoky2_n=prutoky2.x

#statsmodel
res_ols=sm.OLS(y, X).fit()
print(res_ols.summary())
prutoky3_n=res_ols.params
prutoky3_s=res_ols.bse

#index pro dataframe, ktery se bude exportovat
prutoky_indexy_modified=[]
for i in np.arange(1, len(prutoky_indexy)+1):
    if prutoky_indexy.loc[i, 'cilova zona']==N+1:
        prutoky_indexy_modified.append('Re'+str(prutoky_indexy.loc[i, 'vychozi zona']))
    else:
        prutoky_indexy_modified.append('R'+str(prutoky_indexy.loc[i, 'vychozi zona'])+
                                        str(prutoky_indexy.loc[i, 'cilova zona']))

#export
prutoky_hodnoty2=pd.DataFrame(np.array([prutoky3_n, prutoky3_s]).T, index=prutoky_indexy_modified,
                            columns=['R', 'uR'])
prutoky_hodnoty2.index.name='ozn'
prutoky_hodnoty2=prutoky_hodnoty2.sort_index()

prutoky_hodnoty2.to_csv('airflows_preurcena.txt', sep=';', decimal=',', float_format='%.8f')

odezva_TD_df=odezva_TD.loc[:, ['jmeno', 'R_d', 'R_d_err']]
odezva_TD_df=odezva_TD_df.reset_index()
odezva_TD_df.index=[odezva_TD_df.loc[:, 'jmeno'], odezva_TD_df.loc[:, 'zona']]
odezva_TD_df.loc[:,'R_d']=unumpy.uarray(odezva_TD_df.loc[:,'R_d'], odezva_TD_df.loc[:,'R_d_err'])
odezva_TD_df=odezva_TD_df.drop(columns=['jmeno', 'zona', 'plyn', 'R_d_err'])

def f(x):
    return '{:.1f}'.format(x)

odezva_TD_df.to_latex('odezvy_TD.tex', decimal=',',formatters=[f], escape=False, column_format='lrr')
