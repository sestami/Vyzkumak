import numpy as np
import pandas as pd
from uncertainties import ufloat

soubory=['a8','a10','a88','a112']
B=[ufloat(0.889, 0.889*0.1), ufloat(1.440, 1.440*0.1), ufloat(1.655, 1.655*0.1), ufloat(0.925, 0.925*0.1)]

def impulzy(data):
    return data.loc[:, 'sum2']+data.loc[:, 'sum3']

#def citlivost(data):
#    return impulzy(data)/data.loc[:, 'radon[Bq/m3]']
#
#def OAR_nejistota_2(data):
#    return np.sqrt(impulzy(data))/citlivost(data)

def OAR_nejistota(data):
    return data.loc[:, 'radon[Bq/m3]']/np.sqrt(impulzy(data))

def completion(velicina, velicina_err):
    united = np.array([ufloat(value, error) for value, error in zip(velicina.values, velicina_err.values)])
    return pd.DataFrame(data=united, index=velicina.index, columns=[velicina.name])

def hodnoty_a_chyby(velicina):
    '''
    Input:
        array(np.ndarray)
    '''
    shape = velicina.shape
    hodnoty = np.reshape(np.array([el.n for el in velicina.flatten()]), shape)
    smerodatne_odchylky = np.reshape(np.array([el.s for el in velicina.flatten()]), shape)
    return hodnoty, smerodatne_odchylky

for i, el in enumerate(soubory):
    df = pd.read_csv(el+'.tab', encoding="ISO-8859-1", sep="\t",index_col='zaznam')
    OAR_err=OAR_nejistota(df)
    OAR_completed=completion(df.loc[:, 'radon[Bq/m3]'], OAR_err)
    OAR_completed=B[i]*OAR_completed
    OAR_n, OAR_s=hodnoty_a_chyby(OAR_completed.values)

    OAR_n=pd.Series(data=np.concatenate(OAR_n),index=df.index, name='radon[Bq/m3]')
    OAR_s=pd.Series(data=np.concatenate(OAR_s),index=df.index, name='radon_err')

    df_new=pd.concat([df['cas'], OAR_n, OAR_s, df['Teplota[C]'], df['Vlhkost[%]']], axis=1)
    df_new.to_csv(el+'_modified.csv')



# radek nize je pro nahravani dat z TERA sond v dynamickeMereni.py
# df_A=pd.read_csv('.csv', date_parser=dateparse, parse_dates=['cas'], index_col='zaznam')
