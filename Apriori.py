# Author: Fabrício G. M. de Carvalho, Ph.D
# Student: Maria Gabriela Reis

# linear algebra
import numpy as np 
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# lendo o arquivo csv
df = pd.read_csv('./jewelry.csv')

# formatando a coluna do produto (ex: "jewelry.ring" -> "ring")
df['jewelry.earring'] = df['jewelry.earring'].apply(lambda x: x if x is np.nan or isinstance(x, int) else str(x).split('.')[1])

# renomeando as colunas necessárias
df.rename(columns={'2018-12-01 11:40:29 UTC': 'Data e hora',
                   '1924719191579951782': 'ID do pedido',
                   'jewelry.earring': 'Produto',
                   'Unnamed: 9':'Gênero'
                  },
          inplace=True, errors='raise')

# selecionando as colunas necessárias e mostrando
cols = ['Data e hora', "ID do pedido", "Produto"]
df = df[cols].sort_values("Data e hora").dropna()

# capturando os dados dos produtos e montando as transações
order_products = []
all_procucts = []

for name, group in df.head(5000).groupby("ID do pedido"):
    products = []
    for product in group["Produto"].values:
        all_procucts.append(product)
        products.append(product) 
    order_products.append(products)
    
# print(set(df['Produto']))

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
        if ar['support'] >= min_sup and ar['confidence'] >= min_conf:
            pruned_ass_rules.append(ar)
    return pruned_ass_rules

# Apriori para associações entre 2 itens
def apriori_2(itemset, bd, min_sup, min_conf):
    ass_rules = []
    ass_rules.append([]) # nível 1 (grandes conjuntos de itens)
    for item in itemset:
        sup = support({item},{item},bd)
        ass_rules[0].append({'rule': str(item), \
                             'support':sup, \
                             'confidence': 1})        
    ass_rules[0] = prune(ass_rules[0],min_sup, min_conf)
    ass_rules.append([]) # nível 2 (associação de 2 itens)
    for item_1 in ass_rules[0]:
        for item_2 in ass_rules[0]:
            if item_1['rule'] != item_2['rule']:
                rule = item_1['rule']+'_'+item_2['rule']
                Ix = {item_1['rule']}
                Iy = {item_2['rule']}
                sup = support(Ix,Iy, bd)
                conf = confidence(Ix, Iy, bd)
                ass_rules[1].append({'rule':rule, \
                                     'support': sup, \
                                     'confidence': conf})
    ass_rules[1] = prune(ass_rules[1],min_sup, min_conf)
    return ass_rules

# mostrando as regras achadas com confiança maior que 0.2
# apriori_2(itemset, transactions, 0.0, 0.2)[0]

# mostrando regras do produto com o próprio produto (ex: anel -> anel)
rules_with_one_product = pd.DataFrame(apriori_2(itemset, transactions, 0.0, 0.2)[0])
# print("Regras com 1 produto")
# rules_with_one_product

# mostrando regras com 2 produtos diferentes (ex: colar -> brinco)
rules_with_more_products = pd.DataFrame(apriori_2(itemset, transactions, 0.0, 0.2)[1])
print("Regras com 2 produtos")
print(rules_with_more_products)