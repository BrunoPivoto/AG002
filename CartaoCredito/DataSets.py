#encoding=utf-8
import MySQLdb as db
import pandas as pd
from sklearn.model_selection import train_test_split

class CREDIT:

    # Dados do banco
    dbConnection = db.connect('localhost', 'root', 'root', 'statlog')
    input = pd.read_sql("select laufkont, laufzeit, moral, verw, hoehe, sparkont, beszeit, rate, famges, buerge, wohnzeit, verm,  alters, weitkred, wohn, bishkred, beruf, pers, telef, gastarb from germancredit", dbConnection)
    output = pd.read_sql("select kredit from germancredit", dbConnection)

    input_treino, input_teste, output_treino, output_teste = train_test_split(input, output, test_size=0.2)

    dbConnection.close()


