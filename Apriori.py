# Author: Fabrício G. M. de Carvalho, Ph.D
# Student: Maria Gabriela Reis

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# lendo o arquivo csv
df = pd.read_csv('./bread_basket.csv')

# selecionando as colunas necessárias e mostrando
cols = ['Transaction', "Item", "date_time"]
df = df[cols].dropna()

# capturando os dados dos produtos e montando as transações
order_products = []
all_procucts = []

for name, group in df.head(5000).groupby("Transaction"):
    products = []
    for product in group["Item"].values:
        all_procucts.append(product)
        products.append(product) 
    order_products.append(set(products))

# usando o algoritmo Apriori na base de dados
itemset = list(set(all_procucts))
transactions = order_products

# Cálculo do suporte
def support(Ix, Iy, bd):
    sup = 0
    for transaction in bd:
        if (Ix.union(Iy)).issubset(transaction):
            sup+=1
    sup = sup/len(bd)
    return sup

# Cálculo de confiança
def confidence(Ix, Iy, bd):
    Ix_count = 0
    Ixy_count = 0
    for transaction in bd:
        if Ix.issubset(transaction):
            Ix_count+=1
            if (Ix.union(Iy)).issubset(transaction):
                Ixy_count += 1
    conf = Ixy_count / Ix_count
    return conf

# Calculando as regras que não entram nos limites de suporte e confiança
def prune(ass_rules, min_sup, min_conf):
    pruned_ass_rules = []
    for ar in ass_rules:
        if ar['Suporte'] >= min_sup and ar['Confiança'] >= min_conf:
            pruned_ass_rules.append(ar)
    return pruned_ass_rules

# Apriori para associações
def apriori_2(itemset, bd, min_sup, min_conf):
    ass_rules = []
    ass_rules.append([]) # nível 1 (grandes conjuntos de itens)
    for item in itemset:
        sup = support({item},{item},bd)
        ass_rules[0].append({'Regra': str(item), \
                             'Suporte':sup, \
                             'Confiança': 1})        
    ass_rules[0] = prune(ass_rules[0],min_sup, min_conf)
    ass_rules.append([]) # nível 2 (associação de 2 itens)
    for item_1 in ass_rules[0]:
        for item_2 in ass_rules[0]:
            if item_1['Regra'] != item_2['Regra']:
                rule = item_1['Regra']+'_'+item_2['Regra']
                Ix = {item_1['Regra']}
                Iy = {item_2['Regra']}
                sup = support(Ix,Iy, bd)
                conf = confidence(Ix, Iy, bd)
                ass_rules[1].append({'Regra':rule, \
                                     'Suporte': sup, \
                                     'Confiança': conf})
    ass_rules[1] = prune(ass_rules[1],min_sup, min_conf)
    return ass_rules

# Mostrando todos os produtos
print("-----------------------------------------------------------")
print("-- Todos os produtos ")
print("-----------------------------------------------------------")
itemset_df = []
for item in itemset:
    itemset_df.append({"Produtos":item})
print(pd.DataFrame(itemset_df))

# mostrando as regras achadas com suporte maior que 0.05 e confiança maior que 0.1
print("-----------------------------------------------------------")
print("-- Regras ")
print("-----------------------------------------------------------")
rules = pd.DataFrame(apriori_2(itemset, transactions, 0.05, 0.1)[1])
print(rules)

# usando a biblioteca apyori
from apyori import apriori
results = list(apriori(transactions, min_support=0.05, min_confidence=0.1))
rules_apyori = []
for result in results:
    rules_apyori.append({"Regra":result.items, "Suporte": result.support, "Confiança":  result.ordered_statistics[0].confidence})

print("-----------------------------------------------------------")
print("-- Usando a biblioteca Apyori")
print("-----------------------------------------------------------")
print(pd.DataFrame(rules_apyori))