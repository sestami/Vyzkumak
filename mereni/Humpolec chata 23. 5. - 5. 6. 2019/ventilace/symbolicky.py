from sympy import Symbol, linsolve, nonlinsolve, solve, simplify, lambdify
import numpy as np
import time
import pandas as pd
from uncertainties import ufloat, unumpy
import datetime
from ipdb import set_trace
'''
DULEZITE:
    neumi vyresit preurcenou soustavu rovnic!!!!
CIL:
    urceni K, k_E

PREDPOKLADY:
    pocet traceru je stejny jako pocet zon

POSTUP:
step 1: inicializace symbolickych promennych predstavujici veliciny
step 2: hledane veliciny
step 3: definovani rovnic popisujici model
step 4: vyreseni soustavy rovnic vzhledem k velicinam K, k_E
step 5: zjednoduseni vysledku (nepovinne)

POZNAMKY:
    pro N=3 to pocita jeste rychle, pro N=4 je to tragedie. Pro toto N
        je jiz nutne delat zaznamy
'''

def completion(velicina, velicina_err):
    united = np.array([ufloat(value, error) for value, error in zip(velicina.to_numpy(), velicina_err.to_numpy())])
    return pd.Series(data=united, index=velicina.index, name=velicina.name)

#vypocet doby mereni
pocatek=datetime.datetime(2019, 5, 23, 12, 00)
konec=datetime.datetime(2019, 6, 5, 10, 45)
rozdil=konec-pocatek
dT=rozdil.total_seconds()/3600 #v hod

def vypocet(soubor):
    odezva_TD = pd.read_csv('odezva_TD_detektoru'+str(soubor)+'.csv',sep=';', decimal=',',index_col=['zona', 'plyn'])
    odezva_TD = odezva_TD.swaplevel().sort_index()
    informace_plyny = pd.read_csv('plyny_informace'+str(soubor)+'.csv',sep=';', decimal=',',index_col='plyn')

    N=len(odezva_TD.index.levels[1]) #pocet zon
    N_plyny=len(odezva_TD.index.levels[0]) #pocet pouzitych indikacnich plynu

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
    R_d    =odezva_TD.loc[:,'R_d'] #N_plyny x N
    R_d_err=odezva_TD.loc[:,'R_d_err'] #N_plyny x N
    p      =odezva_TD.loc[:,'p']   #N_plyny x N
    T      =odezva_TD.loc[:,'T']+273.15 #N_plyny x N
    odpary =odezva_TD.loc[:,'odpary'] #N_plyny x N

    R_d=completion(R_d, R_d_err)

    U_R   =informace_plyny['U_R'] #N_plyny
    M_w   =informace_plyny['M_w'] #N_plyny
    #vse to proste musi mit rozmer stejny jako C

    R_konst=8.314462618 #univerzalni plynova konstanta [J/K/mol]

    #ZADANI RELATIVNICH NEJISTOT VELICIN M, p, U_R, T
    M_err=0.052
    p_err=0.041
    U_R_err=0.0545
    T_err=0.048

    p  =completion(p, p_err*p)
    U_R=completion(U_R, U_R_err*U_R)
    T  =completion(T, T_err*T)

    #koncentrace a emise
    C=np.full((N_plyny, N), np.nan, dtype=object)
    for k in np.arange(1, N_plyny+1):
        for i in np.arange(1,N+1):
            C[k-1,i-1]=R_d[k,i]/(U_R[k]*dT*60)*M_w[k]*p[k,i]/(R_konst*1000*T[k,i]) # R_d[k, i], p[k, i], T[k, i], U_R[k], M_w[k], dT*60)

    C_hodnoty=pd.DataFrame(C, index=R_d.index.levels[0], columns=R_d.index.levels[1])
    M_hodnoty=odpary/dT
    M_hodnoty=completion(M_hodnoty, M_err*M_hodnoty)

    #SYMBOLICKA CAST
    tic=time.time()
    #inicializace symbolickych promennych predstavujici veliciny
    C=[]
    K=[]
    k_E=[]
    M=np.zeros((N_plyny, N), dtype=object)
    rozmisteni_plynu=np.array([1, 2, 3]) #vzestupne podle ID zony
    #TCE, TMH, MCH, MDC, PCH, PCE

    for k in np.arange(1,N_plyny+1):
    # od jedne do N+1 (pocita se i venkovni prostredi)
        for i in np.arange(1,N+1):
            C.append(Symbol('C_'+str(k)+str(i)))

    for k, i in enumerate(rozmisteni_plynu, 1):
        M[k-1, i-1]=Symbol('m'+str(k)+str(i))

    for i in np.arange(1,N+1):
        for j in np.arange(1,N+1):
            if i==j:
                K.append(0)
            else:
                K.append(Symbol('R'+str(i)+str(j)))
        k_E.append(Symbol('Re'+str(i)))

    C=np.reshape(C, (N_plyny,N))
    K=np.reshape(K, (N,N))
    k_E=np.array(k_E)

    #hledane veliciny
    indexes=K.flatten()!=0
    nezname=K.flatten()[indexes]
    nezname=np.append(nezname, k_E)

    #definovani rovnic popisujici model
    eqn=[]
    for i in np.arange(N):
        for k in np.arange(N_plyny):
            rce=(sum(C[k, :]*K[:,i])-C[k, i]*sum(K[i,:])-C[k,i]*k_E[i]+M[k,i])
            eqn.append(rce)

    #vyreseni soustavy rovnic vzhledem k velicinam K, k_E
    # reseni=solve(eqn, list(nezname), simplify=False)
    reseni=solve(eqn, list(nezname), simplify=False, rational=False)

    #zjednoduseni vysledku
    # reseni=[simplify(reseni[i]) for i in reseni]

    #prevedeni na lambda fce
    prutoky_fce=[lambdify([C.flatten(), M[np.nonzero(M)]], reseni[i]) for i in reseni]
    # C_hodnoty=[koncentrace_vypocet(R_d[i], U_R[i], M_w[i], p[i], T[i], dT*60) for i in np.arange(len(R_d))]

    #pozn: len(odpary) musi byt rovno N_plyny, to je potreba MODIFIKOVAT!
    # M_hodnoty=[emise_vypocet(odpary[i], dT) for i in np.arange(len(odpary))]

    #EXPORT
    informace_plyny=informace_plyny.sort_index()
    pouzite_plyny=informace_plyny.loc[:, 'jmeno']

    prutoky_hodnoty=[el(C_hodnoty.to_numpy().flatten(), M_hodnoty.to_numpy()[np.nonzero(M_hodnoty.to_numpy())]) for el in prutoky_fce]
    prutoky_hodnoty=pd.DataFrame(np.array([unumpy.nominal_values(prutoky_hodnoty), unumpy.std_devs(prutoky_hodnoty)]).T, index=nezname,
                                columns=['R', 'uR'])
    prutoky_hodnoty.index.name='ozn'
    prutoky_hodnoty.to_csv('airflows'+str(soubor)+'.txt', sep=';', decimal=',', float_format='%.8f')

    pouzite_plyny.to_csv('airflows_ID'+str(soubor)+'.csv', header=False, sep=';')
    toc=time.time()
    print('Uplynuly cas: '+str(toc-tic))
    return prutoky_hodnoty

ID_soubory=np.arange(1,9)
for el in ID_soubory:
    prutoky_hodnoty=vypocet(el)

# vypocet('_nepreurcena')

# reseni_N3=[m_22*(C_12*C_33 - C_13*C_32)/(C_11*C_22*C_33 - C_11*C_23*C_32 - C_12*C_21*C_33 + C_12*C_23*C_31 + C_13*C_21*C_32 - C_13*C_22*C_31)]
 # -m_33*(C_12*C_23 - C_13*C_22)/(C_11*C_22*C_33 - C_11*C_23*C_32 - C_12*C_21*C_33 + C_12*C_23*C_31 + C_13*C_21*C_32 - C_13*C_22*C_31),
 # m_11*(C_21*C_33 - C_23*C_31)/(C_11*C_22*C_33 - C_11*C_23*C_32 - C_12*C_21*C_33 + C_12*C_23*C_31 + C_13*C_21*C_32 - C_13*C_22*C_31),
 # m_33*(C_11*C_23 - C_13*C_21)/(C_11*C_22*C_33 - C_11*C_23*C_32 - C_12*C_21*C_33 + C_12*C_23*C_31 + C_13*C_21*C_32 - C_13*C_22*C_31),
 # -m_11*(C_21*C_32 - C_22*C_31)/(C_11*C_22*C_33 - C_11*C_23*C_32 - C_12*C_21*C_33 + C_12*C_23*C_31 + C_13*C_21*C_32 - C_13*C_22*C_31),
 # m_22*(C_11*C_32 - C_12*C_31)/(C_11*C_22*C_33 - C_11*C_23*C_32 - C_12*C_21*C_33 + C_12*C_23*C_31 + C_13*C_21*C_32 - C_13*C_22*C_31),
 # (C_12*C_23*m_33 - C_12*C_33*m_22 - C_13*C_22*m_33 + C_13*C_32*m_22 + C_22*C_33*m_11 - C_23*C_32*m_11)/(C_11*C_22*C_33 - C_11*C_23*C_32 - C_12*C_21*C_33 + C_12*C_23*C_31 + C_13*C_21*C_32 - C_13*C_22*C_31),
 # (-C_11*C_23*m_33 + C_11*C_33*m_22 + C_13*C_21*m_33 - C_13*C_31*m_22 - C_21*C_33*m_11 + C_23*C_31*m_11)/(C_11*C_22*C_33 - C_11*C_23*C_32 - C_12*C_21*C_33 + C_12*C_23*C_31 + C_13*C_21*C_32 - C_13*C_22*C_31),
 # (C_11*C_22*m_33 - C_11*C_32*m_22 - C_12*C_21*m_33 + C_12*C_31*m_22 + C_21*C_32*m_11 - C_22*C_31*m_11)/(C_11*C_22*C_33 - C_11*C_23*C_32 - C_12*C_21*C_33 + C_12*C_23*C_31 + C_13*C_21*C_32 - C_13*C_22*C_31)]
