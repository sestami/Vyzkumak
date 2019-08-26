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

def calculation_OAR_rucne(K, Q):
    K=K[:, :-1]
    K=unumpy.matrix(K)
    K_inverse=K.I
    return -np.dot(K_inverse, Q)

def export_Q(Q, podlazi, airflows_combination):
    def sloupce(patro):
        return r'$Q_'+str(patro)+r'$ $\left[\si{\frac{Bq}{m^3\cdot hod}}\right]$'
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
    dfQ.to_latex('vysledky_Q_rovnovazne_CANARY.tex', decimal=',', formatters=len(podlazi)*[f],  escape=False)
    # dfQ.to_latex('vysledky_OAR.tex', decimal=',', formatters=len(podlazi)*[f],  escape=False)
    return 0

def calculation_infiltrations(P, exfiltrace, ridici_index):
    P=P[:-1, :-1]
    i=ridici_index-1
    infiltrace=exfiltrace+sum(P[i,:]-P[:,i])
    return infiltrace

def calculation_prutoky_ze_zony(N, ridici_index, a, P, V, W):
    P_zaloha=P
    P=P[:-1, :]
    k=np.full(len(a), np.nan, dtype=object)

    pocet_neznamych=len(k)
    index1=[ridici_index]*N
    index2=[]
    for j in np.arange(1,N+2):
        if j!=ridici_index:
            index2.append(j)

    #indexovani prutoku a jejich poloha v vektoru reseni, tj. v podstate takova funkce, mapovani
    prutoky_indexy=pd.DataFrame(np.array([index1, index2]).T, index=np.arange(1,pocet_neznamych+1),
                                columns=['vychozi zona', 'cilova zona'])

    Z=pd.DataFrame(np.zeros((N, pocet_neznamych)),
                columns=np.arange(1, pocet_neznamych+1), dtype=object)
    Z.index+=1

    for m in np.arange(1, pocet_neznamych+1):
        i, j=prutoky_indexy.loc[m]
        if i!=j:
            Z.loc[i, m]=-a[i-1]
            if j<=N:
                Z.loc[j, m]=a[i-1]
    X=Z
    # print(det(unumpy.nominal_values(X)))

    l=ridici_index-1
    y=np.full(len(a), np.nan, dtype=object)
    for i in np.arange(len(y)):
        if i==l:
            y[i]=-sum(a[:]*P[:, i])+prem_kons_Rn*a[i]*V[i]-W[i]
        else:
            # y[i]=a[i]*sum(P[i, :])-sum(a[:l]*P[:l, i])-sum(a[l+1:]*P[l+1:, i])+prem_kons_Rn*a[i]*V[i]-W[i]
            y[i]=a[i]*sum(P[i, :])-sum(a[:]*P[:, i])+a[l]*P[l, i]+prem_kons_Rn*a[i]*V[i]-W[i]

    # X=unumpy.matrix(X)
    # X_inverse=X.I
    # prutoky1=np.dot(X_inverse,y)

    pom=unumpy.matrix(np.dot(X.T, X))
    I2=pom.I
    prutoky=np.dot(I2, np.dot(X.T, y)) #toto je s nejistotami
    prutoky=np.array(prutoky)[0]

    # res_ols=sm.OLS(unumpy.nominal_values(y), unumpy.nominal_values(X)).fit()
    # print(res_ols.summary())
    # vysledek=solve(unumpy.nominal_values(X), unumpy.nominal_values(y))

    prutoky=np.append(prutoky, calculation_infiltrations(P_zaloha, prutoky[-1], ridici_index))
    return prutoky

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

def run(airflows_ID, a_out=0):
    N, P, K, a, V, podlazi = load_data(airflows_ID)
    Q, Q_kovariance, Q_korelace = calculation_Q_conventional(K, a_out, a)
    airflows_combination=load_ID_plynu(airflows_ID)
    return airflows_combination, Q

#SKRIPTOVA CAST
# musime zadat nenulovou koncentraci vnejsiho prostredi, protoze jinak nam to
#nevypocita infiltrace
a_out = 0
# airflows_ID=np.arange(1,9)
airflows_ID=[3]
N, P, K, a, V, podlazi = load_data(airflows_ID[0])

Q_zdroje=unumpy.uarray([400, 114, 0], [51, 13, 0])

airflows_combination_list, Q_list=[], []
OAR_list=[]
for el in airflows_ID:
    N, P, K, a, V, podlazi = load_data(el)
    OAR_list.append(np.array(calculation_OAR_rucne(K, Q_zdroje))[0])
    airflows_combination, Q=run(el)
    airflows_combination_list.append(airflows_combination)
    Q_list.append(Q)
export_Q(OAR_list, podlazi, airflows_combination_list)


W1=4.330*3600
W2=4.010*3600
W=np.array([W1, W2, 0])

#pouze exfiltrace a infiltrace
# exfiltrace_zNamereneho=calculation_exfiltrations(a, a_out, P, V, V*Q)
# exfiltrace=calculation_exfiltrations(a, a_out, P, V, W)

#po zonach
prutoky_Namerene1=calculation_prutoky_ze_zony(N, 1, a, P, V, Q*V)
prutoky_Namerene2=calculation_prutoky_ze_zony(N, 2, a, P, V, Q*V)
prutoky_Namerene3=calculation_prutoky_ze_zony(N, 3, a, P, V, Q*V)

prutoky1=calculation_prutoky_ze_zony(N, 1, a, P, V, W)
prutoky2=calculation_prutoky_ze_zony(N, 2, a, P, V, W)
prutoky3=calculation_prutoky_ze_zony(N, 3, a, P, V, W)
prutoky=[prutoky1, prutoky2, prutoky3]

namerene=[]
for i in np.arange(len(P)-1):
    pom=np.append(P[i, :], P[-1, i])
    namerene.append(pom[pom!=0])
    export_prutoky([prutoky[i], namerene[i]], N, ['zpětně', 'měření'], i+1)
