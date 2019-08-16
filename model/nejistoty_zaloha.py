#lambdify je uzitecne
from sympy import Symbol, diff
import numpy as np
import time

# prem_kons_Rn = np.log(2)/(3.82*24)
prem_kons_Rn = Symbol('\lambda')
#UCENI SE
# x, x1, x2 = symbols('x x1 x2')
# L=Symbol("L")
# eqn=(x*x1*(L**1))+(x2*(L**0))
# s = solve(eqn,L)
# f = lambdify([x, x1, x2], s)#(ufloat(10,0.2), ufloat(8,0.01), ufloat(25,2))

# x, x1, x2 = ufloat(10,0.2), ufloat(25,2), ufloat(8,0.01)

# print(f(x, x1, x2))

# def zkouska(x,y):
    # return x+y**(0.5)+x**2

# print(zkouska(x, x1))

# y, y1, y2 = symbols('y y1 y2')
# K = Symbol("K")
# eqn2=(y*(K**2))+(y1*(K*1))+(y2*(K**0))
# s2 = solve(eqn2,K)
# f2 = lambdify([y, y1, y2], s2)#(ufloat(10,0.2), ufloat(8,0.01), ufloat(25,2))

'''
step 1: definovat promenne a jejich kovariance
step 2: definovat rovnici
step 3: vyresit ji symbolicky
step 4: lambdify
-pozn.: step 3 nebude potreba, uz ji mam vyresenou; toto je obecny postup,
ktery byl pouzit vyse, v "uceni se"
-pozn.: metoda expr.subs() je take hodne uzitecna, spolu s expr.evalf()
-pro jednoduchost zatim uvazuji tri zony
-uvazuji prutoky, vymeny budu uvazovat pozdeji
'''
N=1 #pocet zon

tic=time.time()
#inicializace namerenych velicin
Q=[]
a=[]
K=[]
V=[]
a_diff=[]
for i in np.arange(1,N+2):
#   od jedne do N+1 (pocita se i venkovni)
    a.append(Symbol('a_'+str(i)))
    for j in np.arange(1,N+2):
        if i==j:
            K.append(0)
        else:
            K.append(Symbol('k_{'+str(i)+str(j)+'}'))
    Q.append(Symbol('Q_'+str(i)))
    V.append(Symbol('V_'+str(i)))
    a_diff.append(Symbol('aDiff_'+str(i)))

Q=np.array(Q[:-1])

a=np.array(a)
V=np.array(V[:-1])
a_diff=np.array(a_diff[:-1])

#definovani vektoru namerenych velicin
veliciny=[a,V,a_diff]
for el in K:
    if el!=0:
        veliciny.append(np.array([el]))
veliciny=np.concatenate(veliciny)
pocet_velicin=len(veliciny) #pocet namerenych vstupnich velicin

K=np.reshape(np.array(K),(N+1,N+1))

#definovani kovariancni matice namerenych velicin
cov_matrix=np.full((pocet_velicin, pocet_velicin), np.nan, dtype=object)
for i, el_i in enumerate(veliciny):
    for j, el_j in enumerate(veliciny):
        if i==j:
            cov_matrix[i,i]=Symbol('\sigma^2('+str(el_i)+')')
        else:
            #nediagonalni prvky jsou nulove
            cov_matrix[i,j]=0

#definovani rovnic popisujici model
eqn=np.array([])
for i, Q_i in enumerate(Q):
    rce=(a_diff[i]+prem_kons_Rn*a[i]+(a[i]*sum(K[i,:])-sum(a[:]*K[:,i]))/V[i]-Q_i)
    eqn=np.append(eqn,rce)

#definovani matice derivaci prisunu radonu podle namerenych velicin
dQdx_matrix=np.full((len(Q), pocet_velicin), np.nan, dtype=object)
for i in np.arange(len(dQdx_matrix)):
    for j, velicina in enumerate(veliciny):
        dQdx_matrix[i,j]=diff(eqn[i],velicina)

#vypocet kovariancni matice prisunu radonu
cov_matrix_Q=np.dot(np.dot(dQdx_matrix, cov_matrix), dQdx_matrix.T) #VYSLEDEK
toc=time.time()
print('spotrebovany cas: '+str(toc-tic))
