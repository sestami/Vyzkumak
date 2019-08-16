from sympy import Symbol, diff
import numpy as np
import time
'''
POSTUP
step 1: inicializace promennych predstavujici veliciny
step 2: definovani kovariancni matice namerenych velicin
step 3: definovani soustavy rovnic popisujici model
step 4: definovani matice derivaci prisunu radonu podle namerenych velicin
step 5: vypocet kovariancni matice prisunu radonu
'''
prem_konstanta_Rn = Symbol('\lambda')
N=3 #pocet zon

tic=time.time()
#inicializace symbolickych promennych predstavujici veliciny
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
    a_diff.append(Symbol('\dot{a_'+str(i)+'}'))

Q=np.array(Q[:-1])
a=np.array(a)
V=np.array(V[:-1])
a_diff=np.array(a_diff[:-1])

#definovani vektoru obsahujici vsechny namerene veliciny
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
    rce=(a_diff[i]+prem_konstanta_Rn*a[i]+(a[i]*sum(K[i,:])-sum(a[:]*K[:,i]))/V[i]-Q_i)
    eqn=np.append(eqn,rce)

#definovani matice derivaci prisunu radonu podle namerenych velicin
dQdx_matrix=np.full((len(Q), pocet_velicin), np.nan, dtype=object)
for i in np.arange(len(dQdx_matrix)):
    for j, velicina in enumerate(veliciny):
        dQdx_matrix[i,j]=diff(eqn[i],velicina)

#vypocet kovariancni matice prisunu radonu
#VYSLEDEK
cov_matrix_Q=np.dot(np.dot(dQdx_matrix, cov_matrix), dQdx_matrix.T)
toc=time.time()
print('spotrebovany cas: '+str(toc-tic))
