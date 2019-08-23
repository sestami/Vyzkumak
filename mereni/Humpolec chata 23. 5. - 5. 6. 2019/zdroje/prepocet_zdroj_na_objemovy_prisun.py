import numpy as np
from uncertainties import ufloat, unumpy

#radonove vydejnosti v Bq/hod
W1=4.330*3600
W2=4.010*3600

V1=ufloat(39,5)
V2=ufloat(127,15)

Q0=W1/V1
Q1=W2/V2

print('Q0='+str(Q0))
print('Q1='+str(Q1))