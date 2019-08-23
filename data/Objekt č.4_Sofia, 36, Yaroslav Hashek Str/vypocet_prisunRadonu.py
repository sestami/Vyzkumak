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
def load_data(V_err_rel=V_err_rel, a_out=False):
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

    dfA = pd.read_csv('concentrations.txt',sep=';',decimal=',',index_col='podlazi')
    dfA = completion(dfA.loc[:, 'a'], dfA.loc[:, 'a_err'])
    a = operace(dfA, operace='M')
    if a_out:
        a = np.append(a, ufloat(a_out, 0.05*a_out))

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
                if a_out:
                    P[i, j]=infiltrace(j+1)
                elif a_out==False:
                    P[i, j]=0
            else:
                P[i, j]=R.loc['R'+str(i+1)+str(j+1),'R']

    #matice K
    if a_out:
        K = np.full((N, N+1), np.nan, dtype=object)
    elif a_out==False:
        K = np.full((N, N), np.nan, dtype=object)
    for i in np.arange(len(K[:, 0])):
        for j in np.arange(len(K[0, :])):
            if i == j:
                if i+1==N+1: #tato podminka je ZBYTECNA, protoze nepocitam prisun radonu pro vnejsi prostredi
                    K[i, i] = -sum(P[i, :])
                else:
                    K[i, i] = -sum(P[i, :])-prem_kons_Rn*V[i]
            else:
                K[i, j] = P[j, i]

    #doplneni infiltraci do R, aby se dala exportovat
    R_i, R_i_index=[], []
    for i in np.arange(1,N+1):
        R_i.append(infiltrace(i))
        R_i_index.append('R'+str(N+1)+str(i))
    df_R_i=pd.DataFrame(R_i, index=R_i_index, columns=['R'])
    R=R.append(df_R_i)
    R=R.sort_index()

    K=K/V[:, None]
    return N, R, K, a, V, podlazi

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

def export_inputData(a, V, podlazi, a_out=0):
    '''
    zatim nezahrnuje P (prutoky), N
    vystupuje zde i infiltrovani (v definici df)
    pokud a_out neni rovno nule, pak je uvazovana infiltrace, kterou ale do tabulky vysledku nechceme
    '''
    def f(x):
        return '{:.0f}'.format(x)

    if a_out:
        a = a[:-1]
    df = pd.DataFrame(np.array([a, V]).T, index=podlazi, columns=['$OAR$ [\si{Bq/m^3}]', '$V$ [\si{m^3}]'])
    df.columns.name = 'podlazi'
    df.index.name = None
    df.to_latex('vysledky_inputData.tex', decimal=',', formatters=[f, f], escape=False)
    return 0

def vymena_vzduchu(R, V, N):
    n=0
    for i in np.arange(1, N+1):
        n+=R.loc['R'+str(i)+str(N+1),'R']
    n=n/sum(V)
    return n

def export_R(R,V,N):
    '''
    export prutoku
    '''
    n=vymena_vzduchu(R, V, N)
    R=R.append(pd.DataFrame([n],index=[r'n $[\si{hod^{-1}}]$'], columns=['R']))

    R.index=R.index.str.replace('R','k')
    R=pd.DataFrame(np.array([unumpy.nominal_values(R), unumpy.std_devs(R)]).T[0], columns=['hodnota $\left[\si{m^3/hod}\right]$', r'$\sigma$'],
                   index=R.index)
    R.to_latex('vysledky_prutoky.tex', decimal=',',float_format='%0.2f', escape=False)
    return R

def export_Q(podlazi, a_out, Q):
    def titulek(patro):
        return r'$Q_'+str(patro)+r'$ $\left[\si{\frac{Bq}{m^3\cdot hod}}\right]$'
    def f(x):
        return '{:.0f}'.format(x)

    columns=[]
    for patro in podlazi:
        columns.append(titulek(patro))
    dfQ=pd.DataFrame(Q, index=a_out, columns=columns)
    dfQ.columns.name = '$OAR_{out}$ [\si{Bq/m^3}]'
    dfQ.index.name = None
    # formatters=[f]
    dfQ.to_latex('vysledky_Q.tex', decimal=',', formatters=len(podlazi)*[f],  escape=False)
    return 0

def run(a_out=False):
    N, P, K, a, V, podlazi = load_data(a_out=a_out)
    Q = calculation_Q_conventional(K, a)
    return Q

#SKRIPTOVA CAST
# musime zadat nenulovou koncentraci vnejsiho prostredi, protoze jinak nam to
#nevypocita infiltrace
N, R, K, a, V, podlazi = load_data(a_out=0)
# export_inputData(a, V, podlazi, a_out=10)
export_R(R,V,N)

# a_out = np.array([0, 5, 10, 20, 30])
# Q = np.array([run(el) for el in a_out])
# export_Q(podlazi, a_out, Q)
