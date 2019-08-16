def export(N, P, a, V, podlazi, Q, a_out=0):
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
    if a_out==False:
        a_n, a_s = a_n, a_s
    if a_out:
        a_n, a_s = a_n[:-1], a_s[:-1]
    df = pd.DataFrame(np.array([a_n, a_s, V_n, V_s, Q_n, Q_s]).T, index=podlazi, columns=['OAR [Bq/m^3]', 'u(OAR)', 'V [m^3]', 'u(V)', 'Q [Bq/(m^3*hod)]', 'u(Q)'])
    df.index.rename('podlazi', inplace=True)
    # def f(x):
        # return '%0.0f' % x
    df.to_csv('vysledky'+str(a_out)+'.csv')
    df.columns.name = df.index.name
    df.index.name = None
    df.to_latex('vysledky'+str(a_out)+'.tex', decimal=',', float_format='%s', header=['$OAR$ [\si{Bq/m^3}]', '$u(OAR)$', '$V$ [\si{m^3}]', '$u(V)$', r'$Q$ $\left[\si{\frac{Bq}{m^3\cdot hod}}\right]$', '$u(Q)$'], escape=False)

    #export prutoku
    dfP = pd.DataFrame(P)
    dfP.index += 1
    dfP.columns += 1
    dfP.to_latex('vysledky_prutoky'+str(a_out)+'.tex', column_format=(len(P)+1)*'l')
    return 0

def run(a_out=False):
    N, P, K, a, V, podlazi = load_data(a_out=a_out)
    Q, Q_kovariance, Q_korelace = calculation_Q_conventional(K, a)
    if type(Q)!=bool:
        export(N, P, a, V, podlazi, Q, a_out=a_out)
    return Q, podlazi

# def plotovani(x, y, zona):
    # fig, ax = plt.subplots()
    # y, y_err=hodnoty_a_chyby(y)
    # ax.errorbar(x, y, yerr=y_err, label=zona, fmt='o')
    # ax.grid(axis='y', which='major')
    # ax.set_xlabel('$a_{out}$ [Bq/m$^3$]')
    # ax.set_ylabel('$Q$ [Bq/m$^3$/hod]')
    # ax.legend()
    # plt.show()

# for i, el in enumerate(Q.T):
    # plotovani(a_out, el, podlazi[i])
