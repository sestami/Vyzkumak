import numpy as np
# from numpy.linalg import solve, lstsq
# from scipy.optimize import nnls, lsq_linear
# import sys
from uncertainties import ufloat, covariance_matrix, correlation_matrix, unumpy
from ipdb import set_trace
import pandas as pd
import logging
# logging.getLogger().setLevel(logging.INFO)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib as mpl
mpl.style.use('default')

barvy=['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

barvy_zesvetlene=[lighten_color(el, 0.3) for el in barvy]


# plt.close('all')

'''
Vstupni soubory:
    airflows.txt
    volumes.txt
    a1_modified.csv, a2_modified.csv, ...
TO DO:
    - kovariacni a korelacni matice prisunu radonu!!!! (2. 7. 2019)
    - metodou solve i lstsq -> ASI NE
    - scipy ma take implementovane least square methods; viz scipy.optimize
POZNAMKY:
    - pokud je infiltrovani == True, pak vektor koncentraci a musi obsahovat i
      koncentraci vnejsiho prostredi, tj. ma rozmer N+1 (old)
    - vstupy fce run(), coz je funkce, ktera se pouziva pro vypocet,
      jsou umisteni_sond a koncentrace vnejsiho prostredi a_out
DULEZITE:
    - airflows.txt nesmi obsahovat n!!!
    - load_data ma input umisteni_sond(list/array),
      ktere obsahuje postupne cisla podlazi, ve kterych byly
      sondy umisteny
'''

# premenova konstanta radonu v hod^(-1)
prem_kons_Rn = np.log(2)/(3.82*24)

# a_err_rel=0.2
V_err_rel=0.2
# print("Rel. chyba všech koncentrací je "+str(a_err_rel)+" %.")
print("Rel. chyba všech objemů je "+str(V_err_rel)+" %.")

def completion(velicina, velicina_err):
    united = np.array([ufloat(value, error) for value, error in zip(velicina.to_numpy(), velicina_err.to_numpy())])
    return pd.DataFrame(data=united, index=velicina.index, columns=[velicina.name])

dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')
def load_A(umisteni_sond, doplneni_chyb=False):
    ''', doplneni_chyb=True
    TATO FUNKCE SLOUZI K DVOU UCELUM, BUD VRACI DATAFRAME NEBO NDARRAY
    umisteni_sond(list/array) ... pole obsahujici podlazi, ve kterych byly TERA
        sondy umisteny (postupne, tj pozice 0 je pro a1, pozice 1 pro a2 atd)
    Output: A(dataframe) ... kazdy radek obsahuje OAR z jedne sondy
    '''
    A=[]
    for i in np.arange(1,len(umisteni_sond)+1):
        # df = pd.read_csv('a'+str(i)+'.tab', encoding="ISO-8859-1",
                            # sep="\t",index_col='záznam', parse_dates=['cas'], date_parser=dateparse)

        df=pd.read_csv('a'+str(i)+'_modified.csv', date_parser=dateparse, parse_dates=['cas'], index_col='zaznam',
                       comment='#')
        # a=a[['cas','radon[Bq/m3]']].copy()
        if doplneni_chyb==True:
            #POUZE PRO TERA SONDY JE URCEN NASLEDUJICI RADEK
            # a=completion(df.loc[:,'radon[Bq/m3]'], 8*np.sqrt(0.125*df.loc[:,'radon[Bq/m3]']))
            a=completion(df.loc[:,'radon[Bq/m3]'], df.loc[:,'radon_err'])
            A.append(a.to_numpy()[:, 0])
        elif doplneni_chyb==False:
            a=df.loc[:, 'radon[Bq/m3]']
            A.append(a.to_numpy())
        # A[:, i]=a
    dfA=pd.DataFrame(A, index=umisteni_sond)

    #prumerovani dat ze sond ze stejne zony
    A = np.full((len(np.unique(umisteni_sond)), dfA.shape[1]), np.nan, dtype=object)
    for i, el in enumerate(np.unique(umisteni_sond)):
        if dfA.loc[el].to_numpy().ndim==2:
            A[i] = dfA.loc[el].to_numpy().mean(axis=0)
        else:
            A[i] = dfA.loc[el].to_numpy()
    return A

def load_Time(umisteni_sond):
    '''
    umisteni_sond(list/array) ... pole obsahujici podlazi, ve kterych byly TERA
        sondy umisteny (postupne, tj pozice 0 je pro a1, pozice 1 pro a2 atd)
    Output: Time(list containing dataframes)
    '''
    Time=[]
    for i in np.arange(1,len(umisteni_sond)+1):
        # df = pd.read_csv('a'+str(i)+'.tab', encoding="ISO-8859-1",
                            # sep="\t",index_col='záznam', parse_dates=['cas'], date_parser=dateparse)
        df=pd.read_csv('a'+str(i)+'_modified.csv', date_parser=dateparse, parse_dates=['cas'], index_col='zaznam',
                       comment='#')
        # a=a[['cas','radon[Bq/m3]']].copy()
        cas=df.loc[:,'cas']
        Time.append(cas)
    # A=pd.DataFrame(A,index=np.append(umisteni_sond, 'out'))
    return Time

def load_data(umisteni_sond, airflows_ID, V_err_rel=V_err_rel):
    '''
    - fce pro nacitani prutoku, objemu i koncentraci
    - Zatim neprumeruju koncentrace namerene z ruznych sond v te same zone, TO
      DODELAT!!! (budu potrebovat)
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
    #NACTENI PRUTOKU
    df = pd.read_csv('airflows'+str(airflows_ID)+'.txt',sep=';', decimal=',',index_col='ozn')
    N = int(np.sqrt(len(df))) #pocet zon
    for i in np.arange(1,N+1):
        df.rename(index={'Re'+str(i): 'R'+str(i)+str(N+1)}, inplace=True)
    R = completion(df.loc[:, 'R'], df.loc[:, 'uR'])

    #NACTENI OBJEMU
    dfV = pd.read_csv('volumes.txt',sep=';',decimal=',',index_col='podlazi')
    dfV = completion(dfV.loc[:, 'V'], V_err_rel*dfV.loc[:, 'V'])
    V = np.array([])
    for i in np.unique(dfV.index.values):
        V = np.append(V, sum(dfV.loc[i].to_numpy()))

    podlazi = np.unique(dfV.index.values)

    #NACTENI KONCENTRACI
    A = load_A(umisteni_sond, doplneni_chyb=True) #A je matice, jejiz radky jsou casove vyvoje OAR jednotlivych zon

    #NACTENI PRUTOKU DO MATICE K
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

    #doplneni infiltraci do R, aby se dala exportovat
    R_i, R_i_index=[], []
    for i in np.arange(1,N+1):
        R_i.append(infiltrace(i))
        R_i_index.append('R'+str(N+1)+str(i))
    df_R_i=pd.DataFrame(R_i, index=R_i_index, columns=['R'])
    R=R.append(df_R_i)

    # K=K/V[:, None]
    return N, R, K, A, V, podlazi

#FUNKCE OKOLO KONCENTRACI (PLOTOVANI ATD)

def load_ID_plynu(airflows_ID):
    df=pd.read_csv('airflows_ID'+str(airflows_ID)+'.csv', sep=';',
                   header=None)
    airflows_combination=tuple(df.loc[:, 1])
    return str(airflows_combination).replace('\'','')

def timedeltas(dates):
    '''
    Input: dates(pd.Series)
    pro vypocet casovych rozdilu
    '''
    prevod=3600*10**9 #prevodni koeficient z ns na hodiny
    casove_rozdily=dates.copy()
    for i in np.arange(1,len(dates)):
        casove_rozdily.iloc[i]=(casove_rozdily.iloc[i]-casove_rozdily.iloc[0]).value/prevod
    casove_rozdily.iloc[0]=0
    return casove_rozdily

def casove_derivace(dates, a, graf=False):
    '''
    a nesmi byt s chybami
    '''
    casove_rozdily=timedeltas(dates)
    cs = CubicSpline(casove_rozdily, a)
    if graf==True:
        fig, ax=plt.subplots()
        ax.plot(casove_rozdily, a, 'x', label='data')
        ax.plot(casove_rozdily, cs(casove_rozdily), label='proklad')
        ax.plot(casove_rozdily, cs(casove_rozdily, 1), label='časová derivace')
        plt.legend()
        plt.grid()
        plt.xlabel('uplynulá doba [hod]')
        plt.ylabel(r'OAR [Bq/m$^3$]')
    return cs(casove_rozdily, 1)

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
def calculation_Q_conventional(K, a_out, a, a_diff):
    '''
    a je vektor,      len(a)=N
    a_diff je vektor, len(a_diff)=N
    '''
    # a=np.append(a, ufloat(a_out, 0.05*a_out))
    a=np.append(a, a_out)
    if kontrola_rozmeru(K, a)==False:
        return False, False, False
    Q = a_diff-np.dot(K, a)
    # Q_covarianceMatrix = covariance_matrix(Q)
    # Q_correlationMatrix = correlation_matrix(Q)
    return Q

def graf_Q(podlazi, Dates, Q, airflows_ID, airflows_combination):
    '''
    udela graf vyvoje Q
    '''
    fig, ax=plt.subplots(figsize=(12,7))
    Q=Q.T
    Q_n, Q_s=hodnoty_a_chyby(Q)
    for i in np.arange(len(Q_n)):
        ax.plot(Dates[i], savgol_filter(Q_n[i],7,3), color=barvy[i], label=str(podlazi[i]))
        # ax.plot(Dates[i],Q_n[i], color=barvy[i], label=str(podlazi[i]))
        ax.fill_between(Dates[i], savgol_filter(Q_n[i]-Q_s[i],7,3), savgol_filter(Q_n[i]+Q_s[i],7,3), color=barvy_zesvetlene[i])
        # ax.fill_between(Dates[i], Q_n[i]-Q_s[i], Q_n[i]+Q_s[i], color=barvy_zesvetlene[i])
    print('Pri delani grafu vyvoje Q byl pouzit Savitzky–Golay filtr s velikosti okna 7 a s fitovanim kubickym polynomem (treti stupen)')
    ax.set_xlabel("$datum$")
    # ax.set_ylabel(r"$Q$ $\left[\frac{Bq}{m^3\cdot hod}\right]$")
    ax.set_ylabel(r"$Q$ $\left[\frac{Bq}{hod}\right]$")
    ax.grid()
    days=mdates.DayLocator()
    ax.xaxis.set_major_locator(days)
    format_data = mdates.DateFormatter('%d. %m. %H:%M')
    ax.xaxis.set_major_formatter(format_data)
    fig.autofmt_xdate()
    ax.legend()
    # fig.suptitle(airflows_combination)
    ax.title.set_text(airflows_combination)
    plt.savefig('prisuny'+str(airflows_ID)+'.png',format='png',dpi=200, bbox_inches='tight')
    plt.show()
    return 0

def export_inputData(V, podlazi):
    '''
    zatim nezahrnuje P (prutoky), N
    vystupuje zde i infiltrovani (v definici df)
    pokud a_out neni rovno nule, pak je uvazovana infiltrace, kterou ale do tabulky vysledku nechceme
    '''
    def f(x):
        return '{:.0f}'.format(x)

    df = pd.DataFrame(V, index=podlazi, columns=[r'$V$ [\si{m^3}]'])
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

def export_R(R,V,N, airflows_ID):
    '''
    export prutoku
    '''
    n=vymena_vzduchu(R, V, N)
    R=R.append(pd.DataFrame([n],index=[r'n $[\si{hod^{-1}}]$'], columns=['R']))

    R.index=R.index.str.replace('R','k')
    R=pd.DataFrame(np.array([unumpy.nominal_values(R), unumpy.std_devs(R)]).T[0], columns=['hodnota $\left[\si{m^3/hod}\right]$', r'$\sigma$'],
                   index=R.index)
    R.to_latex('vysledky_prutoky'+str(airflows_ID)+'.tex', decimal=',',float_format='%0.2f', escape=False)
    return R

def export_Q_statistiky(Q, podlazi, airflows_ID):
    '''
    zatim nezahrnuje P (prutoky), N
    vystupuje zde i infiltrovani (v definici df)
    pokud a_out neni rovno nule, pak je uvazovana infiltrace, kterou ale do tabulky vysledku nechceme
    '''
    def titulek(patro):
        return r'$Q_'+str(patro)+r'$ $\left[\si{\frac{Bq}{hod}}\right]$'
    def f(x):
        return '{:.0f}'.format(x)
    columns=[titulek(el) for el in podlazi]
    Q=pd.DataFrame(unumpy.nominal_values(Q), columns=columns)
    statistiky=Q.describe()
    statistiky.to_latex('vysledky_Q_statistiky'+str(airflows_ID)+'.tex', float_format='%0.0f', decimal=',', escape=False)
    return 0

def absolutni_prisuny(A_diff, V):
    print('Pocitaji se absolutni prisuny radonu')
    A_diff_modified=np.full(A_diff.shape, np.nan, dtype=object)
    for i in np.arange(len(A_diff)):
        A_diff_modified[i]=A_diff[i]*V[i]
    return A_diff_modified

def objemove_prisuny(K, V):
    print('Pocitaji se objemove prisuny radonu')
    return K/V[:, None]

def run(umisteni_sond, airflows_ID, a_out=0):
    N, R, K, A, V, podlazi = load_data(umisteni_sond, airflows_ID)
    Dates = load_Time(umisteni_sond)
    A_diff = np.array([casove_derivace(dates, a) for dates, a in zip(Dates, load_A(umisteni_sond, doplneni_chyb=False))])

    #absolutni prisuny
    A_diff=absolutni_prisuny(A_diff, V)
    # objemove prisuny
    # K=objemove_prisuny(K, V)

    Q = np.array([calculation_Q_conventional(K, a_out, a, a_diff) for a, a_diff in zip(A.T, A_diff.T)])
    airflows_combination=load_ID_plynu(airflows_ID)
    graf_Q(podlazi, Dates, Q, airflows_ID, airflows_combination)
    export_R(R, V, N, airflows_ID)
    export_Q_statistiky(Q, podlazi, airflows_ID)
    # Q_covariance = np.array([covariance_matrix(q) for q in Q])
    # Q_correlation = np.array([correlation_matrix(q) for q in Q])
    return N, podlazi, Dates, A, A_diff, V, R, Q, airflows_combination

#SKRIPTOVA CAST

airflows_ID=np.arange(1, 9)

umisteni_sond=[0, 1, 1, 2]
a_out=5

def modifying_R(R, V, N):
    '''
    fce pro pridani vymeny vzduchu do pole s prutoky
    '''
    n=vymena_vzduchu(R, V, N)
    n_df=pd.Series(n,name='n [hod$^{-1}$]',index=R.columns)
    R=R.append(n_df)
    R.columns=[airflows_combination]
    return R

#export Q a R (prutoky a vymena vzduchu)
Q_means_list=[]
airflows_combination_list=[]
N, podlazi, Dates, A, A_diff, V, R, Q, airflows_combination=run(umisteni_sond, airflows_ID[0], a_out=a_out)
Q_means_list.append([Q[:,0].mean(), Q[:,1].mean(), Q[:,2].mean()])
airflows_combination_list.append(airflows_combination)

R=modifying_R(R, V, N)

R_df=pd.DataFrame(R)
for el in airflows_ID[1:]:
    N, podlazi, Dates, A, A_diff, V, R, Q, airflows_combination=run(umisteni_sond, el, a_out=a_out)
    # Q0,Q1,Q2=unumpy.nominal_values(Q[:,0]), unumpy.nominal_values(Q[:,1]), unumpy.nominal_values(Q[:,2])
    Q0,Q1,Q2=Q[:,0], Q[:,1], Q[:,2]
    Q_means_list.append([Q0.mean(), Q1.mean(), Q2.mean()])
    airflows_combination_list.append(airflows_combination)

    R=modifying_R(R, V, N)
    R_df=R_df.join(R)

def f(x):
    return '{:.3f}'.format(x)
def g(x):
    return '{:.0f}'.format(x)

R_df.index=R_df.index.str.replace('R','k')
R_df.to_latex('vysledky_prutoky_CELKOVE.tex', decimal=',',formatters=[f]*len(airflows_ID), escape=False)

Q_df=pd.DataFrame(Q_means_list, index=airflows_combination_list, columns=[r'$Q_0$ $\left[\si{\frac{Bq}{hod}}\right]$',
                                                                          r'$Q_1$ $\left[\si{\frac{Bq}{hod}}\right]$',
                                                                          r'$Q_2$ $\left[\si{\frac{Bq}{hod}}\right]$'])
Q_df.to_latex('vysledky_Q_dynamicky.tex', formatters=3*[g], escape=False)

export_inputData(V, podlazi)
plt.close('all')
# aa=casove_derivace(Dates[0], load_A(umisteni_sond, False).to_numpy()[0], graf=True)

#TOTO BYLO PRO ROVNOVAZNE MERENI
# N, P, K, a, V, podlazi = load_data(a_out=10)
# export_inputData(a, V, podlazi, a_out=10)
# export_P(P, podlazi)

# a_out = np.array([0, 5, 10, 20, 30])
# Q = np.array([run(el) for el in a_out])
# export_Q(podlazi, a_out, Q)
