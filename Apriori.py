import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from mlxtend.frequent_patterns import association_rules, apriori

import warnings
warnings.filterwarnings('ignore')

# Acessando os dados pelo csv
data = pd.read_csv("./bread_basket.csv")
data['date_time'] = pd.to_datetime(data['date_time'])
data['time'] = data['date_time'].dt.time
data.drop('date_time', axis = 1, inplace = True)
data.head(5)


# Análises de dados (sem IA)
# Mesmo sem utilizar inteligências artificiais conseguimos obter análises ricas dos dados que seriam
# o suficiente para já melhorar o serviço do estabelecimento, como os gráficos abaixo:

plt.figure(figsize=(10,5))
sns.barplot(x = data.Item.value_counts().head(5).index, y = data.Item.value_counts().head(5).values, color='pink')
plt.xlabel('Items', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size = 15)
plt.title('Top 5 Items purchased by customers', color = 'black', size = 20)
# plt.show()

plt.figure(figsize=(10,5))
sns.barplot(x = data.period_day.value_counts().index, y = data.period_day.value_counts().values, color='pink')
plt.xlabel('Period', size = 15)
plt.ylabel('Orders per period', size = 15)
plt.title('Number of orders received in each period of a day', color = 'green', size = 20)
# plt.show()

df = data.groupby(['period_day','Item'])['Transaction'].count().reset_index().sort_values(['period_day','Transaction'],ascending=False)
day = ['morning','afternoon','evening','night']

plt.figure(figsize=(15,8))
for i,j in enumerate(day):
    plt.subplot(2,2,i+1)
    df1 = df[df.period_day==j].head(10)
    sns.barplot(data=df1, y=df1.Item, x=df1.Transaction, color='pink')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Top 10 items people like to order in "{}"'.format(j), size=13)

# plt.show()


# Utilizando o APRIORI
df = data.groupby(['Transaction','Item'])['Item'].count().reset_index(name='Count')
my_basket = df.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)

def encode(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
    
my_basket_sets = my_basket.applymap(encode)
frequent_itemsets = apriori(my_basket_sets, min_support = 0.01, use_colnames = True)
# frequent_itemsets

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.sort_values('confidence', ascending = False, inplace = True)
rules