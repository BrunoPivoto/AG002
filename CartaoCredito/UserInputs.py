import numpy as np

class DADOS:
    dados = np.arange(20)
    print("Entre com os dados")
    dados[0] = input("Status da conta corrente (1-4): ")
    dados[1] = input("Duração do crédito em meses: ")
    dados[2] = input("Histórico de cumprimento de contratos (0 - 4): ")
    dados[3] = input("Objetivo (0-10): ")
    dados[4] = input("Valor do crédito: ")
    dados[5] = input("Poupança (1-5): ")
    dados[6] = input("Duração do emprego (1-5): ")
    dados[7] = input("Crédito das prestações como uma porcentagem da renda disponível (1-3): ")
    dados[8] = input("Personal status (1-4): ")
    dados[9] = input("Existe outro devedor ou fiador para o crédito? (1-3) ")
    dados[10] = input("Período de tempo (em anos) em que o devedor vive na residência atual (1-4): ")
    dados[11] = input("Propriedade (1-4): ")
    dados[12] = input("Idade: ")
    dados[13] = input("Planos de parcelamento (1-3): ")
    dados[14] = input("Tipo de habitação em que vive o devedor (1-3): ")
    dados[15] = input("Quantidade de créditos (1-4): ")
    dados[16] = input("Qualidade do trabalho do devedor (1-4): ")
    dados[17] = input("Numero de dependentes financeiros (1-2):  ")
    dados[18] = input("Existe telefone fixo cadastrado em nome do devedor? (1-2) ")
    dados[19] = input("O devedor é um trabalhador estrangeiro? (1-2) ")
