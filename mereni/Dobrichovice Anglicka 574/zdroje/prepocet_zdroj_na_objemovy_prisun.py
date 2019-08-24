import numpy as np
from uncertainties import ufloat, unumpy

#radonove vydejnosti v Bq/hod
W1=4.330*3600
W2=4.010*3600

V0=ufloat(66,13)

Q0=(W1+W2)/V0

print('Q0='+str(Q0))
