import numpy as np
from uncertainties import ufloat, covariance_matrix, correlation_matrix, unumpy
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib as mpl
mpl.style.use('default')
'''
Vstupni soubory:
    airflows.txt
    volumes.txt
    a1_modified.csv, a2_modified.csv, ...
DULEZITE:
    - airflows.txt nesmi obsahovat vymenu vzduchu
'''

def lighten_color(color, amount=0.5):
    """
    source: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

barvy=['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
barvy_zesvetlene=[lighten_color(el, 0.3) for el in barvy]

# premenova konstanta radonu v hod^(-1)
prem_kons_Rn = np.log(2)/(3.82*24)

# relativni chyba vsech objemu
V_err_rel=0.2
print("Rel. chyba vsech objemu je "+str(V_err_rel)+" %.")

def completion(velicina, velicina_err):
    united = np.array([ufloat(value, error) for value, error in zip(velicina.to_numpy(), velicina_err.to_numpy())])
    return pd.DataFrame(data=united, index=velicina.index, columns=[velicina.name])

dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')
def load_A(umisteni_sond, doplneni_chyb=False):
    '''
    Input:
        umisteni_sond(list/array) ... pole obsahujici podlazi, ve kterych byly TERA
            sondy umisteny (postupne, tj pozice 0 je pro a1, pozice 1 pro a2 atd)
    Output:
        A(dataframe) ... kazdy radek obsahuje OAR z jedne sondy
    '''
    A=[]
    for i in np.arange(1,len(umisteni_sond)+1):
        df=pd.read_csv('a'+str(i)+'_modified.csv', date_parser=dateparse, parse_dates=['cas'], index_col='zaznam',
                       comment='#')
        # a=a[['cas','radon[Bq/m3]']].copy()
        if doplneni_chyb==True:
            #POUZE PRO TERA SONDY JE URCEN NASLEDUJICI RADEK
            a=completion(df.loc[:,'radon[Bq/m3]'], df.loc[:,'radon_err'])
            A.append(a.to_numpy()[:, 0])
        elif doplneni_chyb==False:
            a=df.loc[:, 'radon[Bq/m3]']
            A.append(a.to_numpy())
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
        df=pd.read_csv('a'+str(i)+'_modified.csv', date_parser=dateparse, parse_dates=['cas'], index_col='zaznam',
                       comment='#')
        cas=df.loc[:,'cas']
        Time.append(cas)
    return Time

def load_data(umisteni_sond, V_err_rel=V_err_rel):
    '''
    fce pro nacitani prutoku, objemu a vytvoreni matice K
    [a]=Bq/m^3
    [V]=m^3
    [R_ij]=m^3/hod
    V_err_rel=0.2*V
    '''
    #NACTENI PRUTOKU
    df = pd.read_csv('airflows.txt',sep=';', decimal=',',index_col='ozn')
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
                if i+1==N+1:
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

    K=K/V[:, None]
    return N, R, K, A, V, podlazi

#FUNKCE OKOLO KONCENTRACI

def timedeltas(dates):
    '''
    fce pro vypocet casovych rozdilu, potrebnych pro vypocet casovych derivaci
    Input: dates(pd.Series)
    '''
    prevod=3600*10**9 #prevodni koeficient z ns na hodiny
    casove_rozdily=dates.copy()
    for i in np.arange(1,len(dates)):
        casove_rozdily.iloc[i]=(casove_rozdily.iloc[i]-casove_rozdily.iloc[0]).value/prevod
    casove_rozdily.iloc[0]=0
    return casove_rozdily

def casove_derivace(dates, a, graf=False):
    '''
    fce pro vypocet casovych derivaci OAR, umoznuje udelat graf, ktery je obsahuje
    '''
    casove_rozdily=timedeltas(dates)
    cs = CubicSpline(casove_rozdily, a)
    if graf==True:
        fig, ax=plt.subplots()
        ax.plot(casove_rozdily, a, 'x', label='data')
        ax.plot(casove_rozdily, cs(casove_rozdily), label='proklad')
        ax.plot(casove_rozdily, cs(casove_rozdily, 1), label='casova derivace')
        plt.legend()
        plt.grid()
        plt.xlabel('uplynula doba [hod]')
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

#VYPOCET Q
def calculation_Q_conventional(K, a_out, a, a_diff):
    '''
    fce pro vypocet Q
    Input:
        K je matice
        a_out je skalar (OAR vnejsiho prostredi)
        a je vektor,      length(a)=N
        a_diff je vektor, length(a_diff)=N
    '''
    a=np.append(a, a_out)
    if kontrola_rozmeru(K, a)==False:
        return False
    Q = a_diff-np.dot(K, a)
    return Q

def graf_Q(podlazi, Dates, Q):
    '''
    fce pro udelani grafu vyvoje Q; pouziva Savitzky-Golay filtr s velikosti okna 7
    a s fitovanim kubickym polynomem
    '''
    fig, ax=plt.subplots(figsize=(12,7))
    Q=Q.T
    Q_n, Q_s=hodnoty_a_chyby(Q)
    for i in np.arange(len(Q_n)):
        ax.plot(Dates[i], savgol_filter(Q_n[i],7,3), color=barvy[i], label=str(podlazi[i]))
        ax.fill_between(Dates[i], savgol_filter(Q_n[i]-Q_s[i],7,3), savgol_filter(Q_n[i]+Q_s[i],7,3), color=barvy_zesvetlene[i])
    ax.set_xlabel("$datum$")
    ax.set_ylabel(r"$Q$ $\left[\frac{Bq}{m^3\cdot hod}\right]$")
    ax.grid()
    days=mdates.DayLocator()
    ax.xaxis.set_major_locator(days)
    format_data = mdates.DateFormatter('%d. %m. %H:%M')
    ax.xaxis.set_major_formatter(format_data)
    fig.autofmt_xdate()
    ax.legend()
    plt.savefig('prisuny.png',format='png',dpi=200, bbox_inches='tight')
    plt.show()
    return 0

#FUNKCE PRO EXPORT DAT DO TABULEK LATEXOVEHO FORMATU
def export_inputData(V, podlazi):
    #fce pro export objemu zon
    def f(x):
        return '{:.0f}'.format(x)

    df = pd.DataFrame(V, index=podlazi, columns=[r'$V$ [\si{m^3}]'])
    df.columns.name = 'podlazi'
    df.index.name = None
    df.to_latex('vysledky_inputData.tex', decimal=',', formatters=[f, f], escape=False)
    return 0

def vymena_vzduchu(R, V, N):
    #fce pro vypocet vymeny vzduchu
    n=0
    for i in np.arange(1, N+1):
        n+=R.loc['R'+str(i)+str(N+1),'R']
    n=n/sum(V)
    return n

def export_R(R,V,N):
    #export prutoku
    n=vymena_vzduchu(R, V, N)
    R=R.append(pd.DataFrame([n],index=[r'n $[\si{hod^{-1}}]$'], columns=['R']))

    R.index=R.index.str.replace('R','k')
    R=pd.DataFrame(np.array([unumpy.nominal_values(R), unumpy.std_devs(R)]).T[0], columns=['hodnota $\left[\si{m^3/hod}\right]$', r'$\sigma$'],
                   index=R.index)
    R.to_latex('vysledky_prutoky.tex', decimal=',',float_format='%0.2f', escape=False)
    return R

def export_Q_statistiky(Q, podlazi):
    #export statistik prisunu radonu
    def titulek(patro):
        return r'$Q_'+str(patro)+r'$ $\left[\si{\frac{Bq}{m^3\cdot hod}}\right]$'
    def f(x):
        return '{:.0f}'.format(x)
    columns=[titulek(el) for el in podlazi]
    Q=pd.DataFrame(unumpy.nominal_values(Q), columns=columns)
    statistiky=Q.describe()
    statistiky.to_latex('vysledky_Q_statistiky.tex', float_format='%0.0f', decimal=',', escape=False)
    return 0

#VYPOCETNI FUNKCE
def run(umisteni_sond, a_out=0):
    '''
    fce vyuzivajici predchozi funkce pro vypocet Q, export dat do tabulek
    a vytvoreni grafu vyvoje Q; umoznuje take vypocitat kovariancni a korelacni matici
    prisunu radonu
    '''
    N, R, K, A, V, podlazi = load_data(umisteni_sond)
    Dates = load_Time(umisteni_sond)
    A_diff = np.array([casove_derivace(dates, a) for dates, a in zip(Dates, load_A(umisteni_sond, doplneni_chyb=False))])

    Q = np.array([calculation_Q_conventional(K, a_out, a, a_diff) for a, a_diff in zip(A.T, A_diff.T)])
    graf_Q(podlazi, Dates, Q)
    export_R(R, V, N)
    export_Q_statistiky(Q, podlazi)
    export_inputData(V, podlazi)
    # Q_covariance = np.array([covariance_matrix(q) for q in Q])
    # Q_correlation = np.array([correlation_matrix(q) for q in Q])
    return N, podlazi, Dates, A, A_diff, V, R, Q

#SKRIPTOVA CAST

umisteni_sond=[0, 1, 2]
a_out=0

N, podlazi, Dates, A, A_diff, V, R, Q=run(umisteni_sond)
