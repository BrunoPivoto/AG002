import numpy as np
import DataSets
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import Perceptron

# Variaveis de treino e teste
entrada = np.array(DataSets.CREDIT.input_treino).reshape(-1, 20)
saida = np.array(DataSets.CREDIT.output_treino).reshape(-1, 1).ravel()
teste_entrada = np.array(DataSets.CREDIT.input_teste).reshape(-1, 20)
teste_saida = np.array(DataSets.CREDIT.output_teste).reshape(-1, 1).ravel()

# Criação do modelo
model = Perceptron(tol=1e-3)
model.fit(entrada, saida)

# Metricas de Avaliação
predict_train = model.predict(entrada)
predict_test = model.predict(teste_entrada)

print(confusion_matrix(saida, predict_train))
print(classification_report(saida, predict_train, zero_division=0))

print(confusion_matrix(teste_saida, predict_test))
print(classification_report(teste_saida, predict_test, zero_division=0))

mse = mean_squared_error(teste_saida, predict_test)
print("MSE:", mse)

# Teste de Predicoes
ok = 0;
for i in teste_saida:
    resposta = model.predict(np.array([teste_entrada[i]]))
    if resposta != np.array([teste_saida[i]]):
        print("Teste falhou!")
        break
    else:
        ok = ok + 1

if ok == len(teste_saida):
    print("Teste OK")

# Predicoes do usuario
import UserInputs

resposta = model.predict(np.array([UserInputs.DADOS.dados]))
if resposta == 1:
    print("Baseado nos dados de entrada, o devedor é um BOM CANDIDATO")
else:
    print("Baseado nos dados de entrada, o devedor é um MAU CANDIDATO")
