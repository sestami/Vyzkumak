import numpy as np
from uncertainties import ufloat, unumpy

#radonove vydejnosti v Bq/hod
W1=4.330*3600

V1=ufloat(47,9)

Q1=W1/V1

print('Q1='+str(Q1))