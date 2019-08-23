import numpy as np
# from numpy.linalg import solve, lstsq
from scipy.optimize import nnls, lsq_linear
import sys
from uncertainties import ufloat, umath, unumpy, covariance_matrix, correlation_matrix
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
                P[i, j]=0
            elif i+1==N+1:
                P[i, j]=infiltrace(j+1)
            else:
                P[i, j]=R.loc['R'+str(i+1)+str(j+1),'R']

    #matice K
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
    return N, P, K, a, V, podlazi

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
    a = np.append(a, ufloat(a_out, 0.05*a_out))
    if kontrola_rozmeru(K, a)==False:
        return False, False, False
    Q = -np.dot(K, a)
    Q_covarianceMatrix = covariance_matrix(Q)
    Q_correlationMatrix = correlation_matrix(Q)
    return Q, Q_covarianceMatrix, Q_correlationMatrix

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

def export_Q(Q, podlazi, airflows_combination):
    def sloupce(patro):
        return r'$Q_'+str(patro)+r'$ $\left[\si{\frac{Bq}{hod}}\right]$'
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
    dfQ.to_latex('vysledky_Q_rovnovazneCANARY.tex', decimal=',', formatters=len(podlazi)*[f],  escape=False)
    return 0

def run(airflows_ID, a_out=0):
    N, P, K, a, V, podlazi = load_data(airflows_ID)
    Q, Q_kovariance, Q_korelace = calculation_Q_conventional(K, a_out, a)
    airflows_combination=load_ID_plynu(airflows_ID)
    return airflows_combination, Q

#SKRIPTOVA CAST
# musime zadat nenulovou koncentraci vnejsiho prostredi, protoze jinak nam to
#nevypocita infiltrace
a_out = 0
airflows_ID=1
N, P, K, a, V, podlazi = load_data(airflows_ID)
airflows_combination, Q=run(airflows_ID)
# export_Q(Q, podlazi, airflows_combination)
