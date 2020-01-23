import numpy as np
# from numpy.linalg import solve, lstsq
from scipy.optimize import nnls, lsq_linear
import sys
from uncertainties import ufloat, unumpy, umath, covariance_matrix, correlation_matrix
from ipdb import set_trace
import pandas as pd
import logging
# logging.getLogger().setLevel(logging.INFO)
import matplotlib.pyplot as plt

'''
ARGUMENTY:
    - pokud je zadan jakykoliv prvni argument, pak je uvazovana infiltrace
    NOPE, tak to neni!!!
PREDPOKLADY:
    - uvazuji steady state, tj . da/dt = 0, a proto v bilancnich rovnicich
      nevystupuje objem jednotlivych zon
    - POUZE PRO N>1!!!!! NOPE!!!
Vstupni soubory:
TO DO:
    - metodou solve i lstsq -> ASI NE
    - symbolicky pomoci sympy (to bude asi nejlepší potom pro šíření nejistot)
      -> DONE
    - scipy ma take implementovane least square methods; viz scipy.optimize
    - 4. kvetna 2019: naimplementovat nacitani objemu a koncentraci zvlast asi,
      protoze az budu mit k dispozici kontinualni mereni koncentraci v jednotlivych
      zonach, tak to jinak udelat nepujde (to nacitani)
POZNAMKY:
    - pokud je infiltrovani == True, pak vektor koncentraci a musi obsahovat i
      koncentraci vnejsiho prostredi, tj. ma rozmer N+1
'''

# premenova konstanta radonu v hod^(-1)
prem_kons_Rn = np.log(2)/(3.82*24)

# if len(sys.argv)>1:
    # infiltrovani=True
# else:
    # infiltrovani=False

def completion(velicina, velicina_err):
    united = np.array([ufloat(value, error) for value, error in zip(velicina.to_numpy(), velicina_err.to_numpy())])
    return pd.DataFrame(data=united, index=velicina.index, columns=[velicina.name])

V_err_rel=0.2
print("Rel. chyba všech objemů je "+str(V_err_rel)+" %.")
def load_data(airflows_ID, V_err_rel=V_err_rel):
    '''
    - fce pro nacitani prutoku, objemu i koncentraci
    [a]=Bq/m^3
    [V]=m^3
    [R_ij]=m^3/hod
    V_err=0.2*V
    a_out_err=0.2*a_out
    - a_out je koncentrace radonu ve vnejsim prostredi
    - pokud je uvazovano infiltrovani, pak ma vektor koncentrace rozmer
      N+1, pokud neni, pak je rozmer vektoru koncentrace N
    - ve vnejsim prostredi zanedbavame premenu radonu (logicky)
    Output:
        N(int)...pocet zon
        [K_ij]=1/hod (je to podelene tim objemem, tj. nasledne [Q]=Bq/m^3/hod
    '''
    #NACTENI KONCENTRACI A OBJEMU
    def operace(df,operace='M'):
        velicina = np.array([])
        for i in np.unique(df.index.values):
            if operace=='M':
                velicina = np.append(velicina, df.loc[i].to_numpy().mean())
            elif operace=='S':
                velicina = np.append(velicina, sum(df.loc[i].to_numpy()))
        return velicina

    dfA = pd.read_csv('concentrations_CANARY.txt',sep=';',decimal=',',index_col='podlazi')
    dfA = completion(dfA.loc[:, 'a'], dfA.loc[:, 'a_err'])
    a = operace(dfA, operace='M')

    dfV = pd.read_csv('volumes.txt',sep=';',decimal=',',index_col='podlazi')
    dfV = completion(dfV.loc[:, 'V'], V_err_rel*dfV.loc[:, 'V'])
    V = operace(dfV, operace='S')

    podlazi = np.unique(dfV.index.values)

    #NACTENI PRUTOKU DO MATICE K
    df = pd.read_csv('airflows'+str(airflows_ID)+'.txt',sep=';', decimal=',',index_col='ozn')
    N = int(np.sqrt(len(df))) #pocet zon
    for i in np.arange(1,N+1):
        df.rename(index={'Re'+str(i): 'R'+str(i)+str(N+1)}, inplace=True)
    R = completion(df.loc[:, 'R'], df.loc[:, 'uR'])

    def infiltrace(i):
        R_i=R.loc['R'+str(i)+str(N+1),'R']
        for j in np.arange(1,N+1):
            if i!=j:
                R_i+=R.loc['R'+str(i)+str(j),'R']-R.loc['R'+str(j)+str(i),'R']
        return R_i

    #matice P
    P = np.full((N+1, N+1), np.nan, dtype=object)
    for i in np.arange(len(P[:, 0])):
        for j in np.arange(len(P[0, :])):
            if i==j:
                P[i, j]=ufloat(0, 0)
            elif i+1==N+1:
                P[i, j]=infiltrace(j+1)
            else:
                P[i, j]=R.loc['R'+str(i+1)+str(j+1),'R']

    return N, P, a, V, podlazi

def calculation_K(N, P, V):
    K = np.full((N, N+1), np.nan, dtype=object)
    for i in np.arange(len(K[:, 0])):
        for j in np.arange(len(K[0, :])):
            if i == j:
                if i+1==N+1: #tato podminka je ZBYTECNA, protoze nepocitam prisun radonu pro vnejsi prostredi
                    K[i, i] = -sum(P[i, :])
                else:
                    K[i, i] = -sum(P[i, :])-prem_kons_Rn*V[i]
            else:
                K[i, j] = P[j, i]

    K=K/V[:, None]
    return K


def load_ID_plynu(airflows_ID):
    df=pd.read_csv('airflows_ID'+str(airflows_ID)+'.csv', sep=';',
                   header=None)
    airflows_combination=tuple(df.loc[:, 1])
    return str(airflows_combination).replace('\'','')

def kontrola_rozmeru(K, a):
    if len(a) != len(K[0, :]):
        logging.error('Neodpovidaji rozmery matice K a vektoru a!!')
        return False
    return True

#doplnujici funkce
def hodnoty_a_chyby(velicina):
    '''
    Input:
        array(np.ndarray)
    '''
    shape = velicina.shape
    hodnoty = np.reshape(np.array([el.n for el in velicina.flatten()]), shape)
    smerodatne_odchylky = np.reshape(np.array([el.s for el in velicina.flatten()]), shape)
    return hodnoty, smerodatne_odchylky

#vypocetni funkce
def calculation_Q_conventional(K, a_out, a):
    #pridani vnejsi koncentrace do vektoru koncentraci
    a = np.append(a, a_out)
    if kontrola_rozmeru(K, a)==False:
        return False, False, False
    Q = -np.dot(K, a)
    # Q_covarianceMatrix = covariance_matrix(Q)
    # Q_correlationMatrix = correlation_matrix(Q)
    return Q

# def zaokrouhleni(value,error):
    # for i, (n, s) in enumerate(zip(value,error)):
        # if s == 0:
            # return 0
        # sgn = -1 if s < 0 else 1
        # scale = int(-np.floor(np.log10(abs(s))))
        # factor = 10**(scale+1)
        # s=sgn*round(abs(s)*factor)/factor
        # sgn = -1 if n < 0 else 1
        # n=sgn*round(abs(n)*factor)/factor
        # value[i], error[i] = n, s
    # return value, error

def calculation_OAR_rucne(K, Q):
    K=K[:, :-1]
    K=unumpy.matrix(K)
    K_inverse=K.I
    return -np.dot(K_inverse, Q)

def export_Q(Q, podlazi, airflows_combination):
    def sloupce(patro):
        # return r'$Q_'+str(patro)+r'$ $\left[\si{\frac{Bq}{m^3\cdot hod}}\right]$'
        return 'Q_'+str(patro)
    def f(x):
        return '{:.0f}'.format(x)

    columns=[]
    for patro in podlazi:
        columns.append(sloupce(patro))
    dfQ=pd.DataFrame(Q, index=airflows_combination, columns=columns)
    # dfQ.columns.name = '$OAR_{out}$ [\si{Bq/m^3}]'
    dfQ.columns.name = None
    dfQ.index.name = None
    # formatters=[f]
    # dfQ.to_latex('vysledky_Q_rovnovazne_CANARY.tex', decimal=',', formatters=len(podlazi)*[f],  escape=False)
    # dfQ.to_csv('vysledky_Q_rovnovazne_CANARY.csv')
    dfQ.to_csv('vysledky_Q_rovnovazne.csv')
    # dfQ.to_latex('vysledky_OAR.tex', decimal=',', formatters=len(podlazi)*[f],  escape=False)
    return 0

def calculation_exfiltrations(a, a_out, P, V, W):
    P=P[:, :-1]
    # y=np.full(len(a), np.nan, dtype=object)
    exfiltrace=np.full(len(a), np.nan, dtype=object)

    # a = np.append(a, ufloat(a_out, 0.05*a_out))

    for i in np.arange(len(exfiltrace)):
        # y[i]=a[i]*sum(P[i,:])-sum(a[:]*P[:, i])+prem_kons_Rn*V[i]*a[i]-W[i]
        exfiltrace[i]=-sum(P[i,:])+sum(a[:]*P[:-1, i])/a[i]-prem_kons_Rn*V[i]+W[i]/a[i]
    return exfiltrace

def calculation_infiltrations(P, exfiltrace, ridici_index):
    P=P[:-1, :-1]
    i=ridici_index-1
    infiltrace=exfiltrace+sum(P[i,:]-P[:,i])
    return infiltrace

def export_prutoky(Q, N, airflows_combination, ridici_index):
    def sloupce(j):
        # return r'$Q_'+str(patro)+r'$ $\left[\si{\frac{Bq}{hod}}\right]$'
        # return r'\multicolumn{2}{r}{$k_{'+str(ridici_index)+str(j)+r'}$ [\si{m^3/hod}]}'
        return r'$k_{'+str(ridici_index)+str(j)+r'}$ [\si{m^3/hod}]'
    def f(x):
        return '{:.2f}'.format(x)

    columns=[]
    for j in np.arange(1,N+1):
        if j!=ridici_index:
            columns.append(sloupce(j))
    columns.append(sloupce('E'))
    columns.append(sloupce('I'))
    dfQ=pd.DataFrame(Q, index=airflows_combination, columns=columns)
    # dfQ=dfQ.T
    # dfQ.columns.name = '$OAR_{out}$ [\si{Bq/m^3}]'
    dfQ.columns.name = None
    dfQ.index.name = None
    # formatters=[f]
    # dfQ.to_latex('vysledky_Q_rovnovazneCANARY.tex', decimal=',', formatters=len(podlazi)*[f],  escape=False)
    formatovani=r'>{\collectcell\num}r<{\endcollectcell}@{${}\pm{}$}>{\collectcell\num}r<{\endcollectcell}'
    dfQ.to_latex('zpetnyChod_prutoky'+str(ridici_index)+'.tex', decimal=',', formatters=(N+1)*[f],
                 column_format='l'+'r'*(N+1),escape=False)
    return 0

def run(N, P, V, a, a_out=0):
    K=calculation_K(N, P, V)
    Q = calculation_Q_conventional(K, a_out, a)
    Q_rel=unumpy.std_devs(Q)/unumpy.nominal_values(Q)
    # airflows_combination=load_ID_plynu(airflows_ID)
    return Q, Q_rel, K

#SKRIPTOVA CAST
a_out = 0
# a_out = ufloat(a_out, a_out*0.05)
def nastaveni_nejistotyNaNulu(velicina):
    shape = velicina.shape
    velicina_new=[]
    for el in velicina.flatten():
        velicina_new.append(ufloat(el.n,0))
    velicina_new=np.reshape(velicina_new, shape)
    return velicina_new

airflows_ID=[3]
N, P, a, V, podlazi = load_data(airflows_ID[0])
P=nastaveni_nejistotyNaNulu(P)
a=nastaveni_nejistotyNaNulu(a)
V=nastaveni_nejistotyNaNulu(V)

def udeleni_relNejistoty_velicine(velicina, rel_chyba):
    velicina=ufloat(velicina, velicina*rel_chyba)
    return velicina

#exfiltrace, resp. prutoky
rel_nejistota=0.5
i,j=2,-1
P[i,j]=udeleni_relNejistoty_velicine(P[i,j].n, rel_nejistota)
Q, Q_rel, K=run(N, P, V, a, a_out)

Q_zdroje=unumpy.uarray([400, 114, 0], [51, 13, 0])
