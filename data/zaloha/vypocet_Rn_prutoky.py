import numpy as np
# from numpy.linalg import solve, lstsq
from scipy.optimize import nnls, lsq_linear
import sys
from uncertainties import ufloat, umath, unumpy, covariance_matrix, correlation_matrix
from ipdb import set_trace
import pandas as pd
import logging

'''
ARGUMENTY:
    - pokud je zadan jakykoliv prvni argument, pak je uvazovana infiltrace
PREDPOKLADY:
    - uvazuji steady state, tj . da/dt = 0, a proto v bilancnich rovnicich
      nevystupuje objem jednotlivych zon
    - POUZE PRO N>1!!!!!
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

if len(sys.argv)>1:
    infiltrovani=True
else:
    infiltrovani=False

def completion(velicina, velicina_err):
    united = np.array([ufloat(value, error) for value, error in zip(velicina.to_numpy(), velicina_err.to_numpy())])
    return pd.DataFrame(data=united, index=velicina.index, columns=[velicina.name])

def load_data(V_err_rel=0.2, infiltrovani=infiltrovani):
    '''
    - fce pro nacitani prutoku, objemu i koncentraci
    [a]=Bq/m^3
    [V]=m^3
    [R_ij]=m^3/hod
    V_err=0.2*V
    - pokud je uvazovano infiltrovani, pak ma vektor koncentrace rozmer
      N+1, pokud neni, pak je rozmer vektoru koncentrace N
    - ve vnejsim prostredi zanedbavame premenu radonu (logicky)
    Output:
        [K_ij]=1/hod (je to podelene tim objemem, tj. nasledne [Q]=Bq/m^3/hod
    '''
    #NACTENI KONCENTRACI A OBJEMU
    def operace(df,operace='N'):
        velicina = np.array([])
        for i in np.unique(df.index.values):
            if operace=='N':
                velicina = np.append(velicina, df.loc[i].to_numpy().mean())
            elif operace=='S':
                velicina = np.append(velicina, sum(df.loc[i].to_numpy()))
        return velicina
    dfA = pd.read_csv('concentrations.txt',sep=';',decimal=',',index_col='podlazi')
    dfA = completion(dfA.loc[:, 'a'], dfA.loc[:, 'a_err'])
    a = operace(dfA, operace='N')
    dfV = pd.read_csv('volumes.txt',sep=';',decimal=',',index_col='podlazi')
    dfV = completion(dfV.loc[:, 'V'], V_err_rel*dfV.loc[:, 'V'])
    V = operace(dfV, operace='S')
    podlazi = np.unique(dfV.index.values)

    #NACTENI PRUTOKU DO MATICE K
    df = pd.read_csv('airflows.txt',sep=';', decimal=',',index_col='ozn')
    N = int(np.sqrt(len(df)-1)) #pocet zon
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
                if infiltrovani==True:
                    P[i, j]=infiltrace(j+1)
                elif infiltrovani==False:
                    P[i, j]=0
            else:
                P[i, j]=R.loc['R'+str(i+1)+str(j+1),'R']
    #matice K
    if infiltrovani==True:
        K = np.full((N, N+1), np.nan, dtype=object)
    elif infiltrovani==False:
        K = np.full((N, N), np.nan, dtype=object)
    for i in np.arange(len(K[:, 0])):
        for j in np.arange(len(K[0, :])):
            if i == j:
                if i+1==N+1:
                    K[i, i] = -sum(P[i, :])
                else:
                    K[i, i] = -sum(P[i, :])-prem_kons_Rn*V[i]
            else:
                K[i, j] = P[j, i]

    K=K/V[:, None]
    return N, P, K, a, V, podlazi

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
def calculation_Q_conventional(K, a):
    if kontrola_rozmeru(K, a)==False:
        return False, False, False
    Q = -np.dot(K, a)
    Q_covarianceMatrix = covariance_matrix(Q)
    Q_correlationMatrix = correlation_matrix(Q)
    return Q, Q_covarianceMatrix, Q_correlationMatrix

def export(N, P, a, V, podlazi, Q):
    '''
    zatim nezahrnuje P (prutoky), N
    vystupuje zde i infiltrovani (v definici df)
    '''
    def zaokrouhleni(value,error):
        for i, (n, s) in enumerate(zip(value,error)):
            if s == 0:
                return 0
            sgn = -1 if s < 0 else 1
            scale = int(-np.floor(np.log10(abs(s))))
            factor = 10**(scale+1)
            s=sgn*round(abs(s)*factor)/factor
            sgn = -1 if n < 0 else 1
            n=sgn*round(abs(n)*factor)/factor
            value[i], error[i] = n, s
        return value, error

    a_n, a_s = zaokrouhleni(*hodnoty_a_chyby(a))
    V_n, V_s = zaokrouhleni(*hodnoty_a_chyby(V))
    Q_n, Q_s = zaokrouhleni(*hodnoty_a_chyby(Q))
    if infiltrovani==False:
        df = pd.DataFrame(np.array([a_n, a_s, V_n, V_s, Q_n, Q_s]).T, index=podlazi, columns=['a [Bq/m^3]', 'u(a)', 'V [m^3]', 'u(V)', 'Q [Bq/hod]', 'u(Q)'])
    if infiltrovani==True:
        df = pd.DataFrame(np.array([a_n[:-1], a_s[:-1], V_n, V_s, Q_n, Q_s]).T, index=podlazi, columns=['a [Bq/m^3]', 'u(a)', 'V [m^3]', 'u(V)', 'Q [Bq/hod]', 'u(Q)'])
    df.index.rename('podlazi', inplace=True)
    # def f(x):
        # return '%0.0f' % x
    df.to_csv('vysledky.csv')
    df.columns.name = df.index.name
    df.index.name = None
    df.to_latex('vysledky.tex', decimal=',', float_format='%s')

    #export prutoku
    dfP = pd.DataFrame(P)
    dfP.index += 1
    dfP.columns += 1
    dfP.to_latex('vysledky_prutoky.tex', column_format=(len(P)+1)*'l')
    return 0

#SKRIPTOVA CAST
N, P, K, a, V, podlazi = load_data()
Q, Q_kovariance, Q_korelace = calculation_Q_conventional(K, a)
if type(Q)!=bool:
    export(N, P, a, V, podlazi, Q)
