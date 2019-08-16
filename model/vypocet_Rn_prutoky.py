import numpy as np
# from numpy.linalg import solve, lstsq
from scipy.optimize import nnls, lsq_linear
import sys
from uncertainties import ufloat, umath, unumpy, covariance_matrix, correlation_matrix
from ipdb import set_trace
import pandas as pd

'''
PREDPOKLADY:
    - uvazuji steady state, tj . da/dt = 0, a proto v bilancnich rovnicich
      nevystupuje objem jednotlivych zon
Vstupni soubory:
    - soubor s matici prutoku vzduchu mezi jednotlivymi zonami a vnejsim
      prostredim, i s nejistotami (tj. soubor obsahuje dve matice)
    - soubor s namerenymi koncentracemi a objemy v jednotlivych zonach s nejistotami
      (tj. soubor obsahuje tabulku, jejiz sloupce jsou jednotlive veliciny a radky
      predstavuji zony)
TO DO:
    - JE POTREBA PREVEST VYMENY VZDUCHU Z hod^(-1) na s^(-1)??? (podle me ne,
      kdyz vse na obou stranach rovnic dosadime ve stejnych jednotkach) -> NE
    - metodou solve i lstsq -> ASI NE
    - symbolicky pomoci sympy (to bude asi nejlepší potom pro šíření nejistot)
      -> DONE
    - scipy ma take implementovane least square methods; viz scipy.optimize
    - 4. kvetna 2019: naimplementovat nacitani objemu a koncentraci zvlast asi,
      protoze az budu mit k dispozici kontinualni mereni koncentraci v jednotlivych
      zonach, tak to jinak udelat nepujde (to nacitani)
'''

# premenova konstanta radonu v hod^(-1)
prem_kons_Rn = np.log(2)/(3.82*24)

def make_P():
    '''
    nacteni matice prutoku vzduchu mezi jednotlivymi zonami
    Vyvoj:
        - vystup je matice napechovana hodnotami s nejistotami (type: ufloat) a
          kovariancni matice
        - mozna jsou ty kontroly zbytecne...
    '''
    P = np.nan
    P_err = np.nan
    if len(sys.argv) > 1:
        data = np.loadtxt(sys.argv[1])
        rozmer = len(data[0,:])
        P = data[:rozmer]
        P_err = data[rozmer:]
        if (P.shape[0] == P.shape[1])and(P_err.shape[0] == P_err.shape[1]):
            if P.diagonal().any():
                print('Diagonalni prvky matice prutoku vzduchu nejsou nulove (tj. nektere k_ii je nenulove).')
                print('Presto jsou brana vstupni data jako relevantni a pocita s nimi.')
            if P.shape[0] != P_err.shape[0]:
                print('Byl zadan jiny pocet hodnot k_ij nez pocet jejich nejistot!')
                P_err = np.nan
        else:
            print('Matice prutoku vzduchu musi byt ctvercova!')
            P = np.nan

    if np.isscalar(P):
        print('Matice prutoku vzduchu nebyla zadana, proto jsou brany defaultni hodnoty.')
        P = np.array([[0, 3.6, 2.9, 40], [80.5, 0, 20.8, 35], [23.7, 0, 0, 20], [2, 60, 25, 0]])
        P_err = np.array([[0, 0.1, 0.2, 4],  [8, 0, 2, 3], [3, 0, 0, 1.5], [0.5, 8, 4, 0]])
    if np.isscalar(P_err):
        print('Nastavuji hodnoty nejistot k_ij na defaultni.')
        P_err = np.array([[0, 0.1, 0.2, 4],  [8, 0, 2, 3], [3, 0, 0, 1.5], [0.5, 8, 4, 0]])
    P_completed = np.array([ufloat(value, error) for value, error in zip(P.flatten(), P_err.flatten())])
    P_completed = np.reshape(P_completed, P.shape)
    return P_completed

def make_K(V, P=make_P()):
    # P = make_P()
    K = np.zeros_like(P)
    for i in np.arange(len(K[:, 0])):
        for j in np.arange(len(K[0, :])):
            if i == j:
                K[i, i] = -sum(P[i, :])+P[i, i]-prem_kons_Rn*V[i]
            else:
                K[i, j] = P[j, i]
    return K

def load_data():
    def completion(velicina, velicina_err):
        return np.array([ufloat(value, error) for value, error in zip(velicina, velicina_err)])

    a = np.nan
    V = np.nan
    if len(sys.argv) > 2:
        data=pd.read_csv(sys.argv[2],sep=' ',index_col='zona')
        # data=np.loadtxt(sys.argv[2])
        a = data.loc[:, 'a[Bq/m3]'].to_numpy()
        a_err = data.loc[:, 'a_err'].to_numpy()
        V = data.loc[:, 'V[m3]'].to_numpy()
        V_err = data.loc[:, 'V_err'].to_numpy()
    if np.isscalar(a) or np.isscalar(V) or (len(a) != len(a_err)) or (len(V) != len(V_err)) or (len(a) != len(V)):
        print('Koncentrace radonu nebo objemy mistnosti nebo nejistoty techto\
velicin nebyly zadany spravne nebo nebyly zadany vubec, proto jsou brany\
defaultni hodnoty. Musi odpovidat rozmerove atd.')
        a = np.array([350, 200, 300, 0])
        a_err = np.array([20, 40, 50, 0])
        V = np.array([100, 250, 200, 0])
        V_err = np.array([20, 50, 40, 0])
    # if len(a) != len(a_err):
        # print('Nebyl zadan stejny pocet chyb jako koncentraci, proto jsou brany defaultni chyby.')
        # a_err = np.array([20, 40, 50, 0])
    a_completed = completion(a, a_err)
    V_completed = completion(V, V_err)
    return a_completed, V_completed

def prepare_for_calculation():
    a, V = load_data()
    K = make_K(V)
    if len(a) != len(K[0, :]):
        print('Neodpovidaji rozmery matice K a vektoru a, resp V!!')
        return np.nan, np.nan
    return K, a

def pouze_nominalni_hodnoty(array):
    '''
    Input:
        array(np.ndarray)
    '''
    shape = array.shape
    return np.reshape(np.array([el.n for el in array.flatten()]), shape)

def calculation_Q_conventional():
    K, a = prepare_for_calculation()
    # breakpoint()
    if np.isscalar(K):
        print('Nelze provest vypocet.')
    Q = -np.dot(K, a)
    Q_covarianceMatrix = covariance_matrix(Q)
    Q_correlationMatrix = correlation_matrix(Q)
    return Q, Q_covarianceMatrix, Q_correlationMatrix

def calculation_Q_nnls():
    K, a = prepare_for_calculation()
    # set_trace()
    K = pouze_nominalni_hodnoty(K)
    a = pouze_nominalni_hodnoty(a)
    if np.isscalar(K):
        print('Nelze provest vypocet.')
    b = -np.dot(K, a)
    Q, residuals = nnls(np.eye(len(K[0, :])), b)
    return Q, residuals

def calculation_Q_lsqLinear():
    '''
    metodou lsq_linear lze nahradit nnls
    nastavil jsem to tak, ze by to melo delat to same jako nnls
    '''
    K, a = prepare_for_calculation()
    K = pouze_nominalni_hodnoty(K)
    a = pouze_nominalni_hodnoty(a)
    if np.isscalar(K):
        print('Nelze provest vypocet.')
    b = -np.dot(K, a)
    result = lsq_linear(np.eye(len(K[0, :])), b, bounds=(0, np.inf))
    return result

# P = make_P()
# K = make_K()
Q_conv, Q_convKovariance, Q_convKorelace = calculation_Q_conventional()
# Q_nnls, residuals = calculation_Q_nnls()
# result_lsqLinear = calculation_Q_lsqLinear()

print()
print('Q_conv = '+str(Q_conv))
# print('Q_nnls = '+str(Q_nnls))
# print('result_lsqLinear = '+str(result_lsqLinear))
