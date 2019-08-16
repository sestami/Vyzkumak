# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:12:11 2019

@author: michal.sestak
"""

from uncertainties import ufloat, umath, unumpy, covariance_matrix
from sympy import solve, symbols, Symbol, lambdify, Eq, diff
import numpy as np
import time

prem_kons_Rn = np.log(2)/(3.82*24)
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
N=3 #pocet zon

tic=time.time()
#definovani velicin
Q=[]
a=[]
K=[]
V=[]
a_diff=[]
for i in np.arange(1,N+2):
    #od jedne do N+1 (pocita se i venkovni)
    a.append(Symbol('a_'+str(i)))
    for j in np.arange(1,N+2):
        K.append(Symbol('k_'+str(i)+str(j)))
#    if i==N+1:
#        Q.append(0)
#        V.append(0)
#        a_diff.append(0)
    Q.append(Symbol('Q_'+str(i)))
    V.append(Symbol('V_'+str(i)))
    a_diff.append(Symbol('aDiff_'+str(i)))

Q=np.array(Q[:-1])

a=np.array(a)
V=np.array(V[:-1])
a_diff=np.array(a_diff[:-1])
K=np.reshape(np.array(K),(N+1,N+1))

veliciny=[a,V,a_diff] #mineno vstupni veliciny, Q je hledana velicina
#veliciny=np.concatenate((veliciny, K))
for el in K:
    veliciny.append(el)
veliciny=np.concatenate(veliciny)


#definovani kovariancni matice vstupnich velicin
pocet_velicin=len(a)+len(V)+len(a_diff)+len(K)*len(K[0,:]) #vstupnich velicin
cov_matrix=np.full((pocet_velicin, pocet_velicin), np.nan, dtype=object) #predelat matici plnou nan!!
for i, el_i in enumerate(veliciny):
    for j, el_j in enumerate(veliciny):
        if j>=i:
            cov_matrix[i,j]=Symbol('s('+str(el_i)+', '+str(el_j)+')')
        # elif j<i:
        else:
            cov_matrix[i,j]=Symbol('s('+str(el_j)+', '+str(el_i)+')')
        # else:
            # print("Chyba! Nejaky element kovariancni matice vstupnich velicin nebyl naplnen.")
        # cov_matrix[i,j]=Symbol('s_')

#definovani kovariancni matice vystupnich velicin (Q_i)
# cov_matrix_Q=np.full((len(Q), len(Q)), np.nan, dtype=object) #predelat matici plnou nan!!
# for i, el_i in enumerate(Q):
    # for j, el_j in enumerate(Q):
        # cov_matrix_Q[i,j]=Symbol('s('+str(el_i)+', '+str(el_j)+')')


#definovani rovnic
#vyreseni rovnic -> zatim ne, zrejme zbytecne
eqn=np.array([])
for i, Q_i in enumerate(Q):
    rce=(prem_kons_Rn*a[i]+a_diff[i]+(a[i]*sum(K[i,:])-sum(a[:]*K[:,i]))/V[i]-Q_i)
    eqn=np.append(eqn,rce)

#definovani matice derivaci vystupnich velicin podle vstupnich velicin
dQdx_matrix=np.full((len(Q), pocet_velicin), np.nan, dtype=object) #predelat matici plnou nan!!
for i in np.arange(len(dQdx_matrix)):
    for j, velicina in enumerate(veliciny):
        dQdx_matrix[i,j]=diff(eqn[i],velicina)

#vypocet kovarianci vystupnich velicin
# for i, dQ_i in enumerate(dQdx_matrix):
    # for j, dQ_j in enumerate(dQdx_matrix):
        # cov_matrix_Q[i, j]=sum(np.dot(diff(Q_i,veliciny),))
cov_matrix_Q=np.dot(np.dot(dQdx_matrix, cov_matrix), dQdx_matrix.T)
toc=time.time()
print('spotrebovany cas: '+str(toc-tic))
