#!/usr/bin/env python
# coding: utf-8

# ## Desafio
# 
# Você foi alocado(a) em um time da Indicium que está trabalhando atualmente junto a um cliente no processo de criação de uma plataforma de aluguéis temporários na cidade de Nova York. Para o desenvolvimento de sua estratégia de precificação, pediu para que a Indicium fizesse uma análise exploratória dos dados de seu maior concorrente, assim como um teste de validação de um modelo preditivo.
# 
# Seu objetivo é desenvolver um modelo de previsão de preços a partir do dataset oferecido, e avaliar tal modelo utilizando as métricas de avaliação que mais fazem sentido para o problema. O uso de outras fontes de dados além do dataset é permitido (e encorajado). Você poderá encontrar em anexo um dicionário dos dados.

# ## Dicionário dos dados
# 
# 	A base de dados de treinamento contém 16 colunas. Seus nomes são auto-explicativos, mas, caso haja alguma dúvida, a descrição das colunas é:
# 
# **id** – Atua como uma chave exclusiva para cada anúncio nos dados do aplicativo
# **nome** - Representa o nome do anúncio
# **host_id** - Representa o id do usuário que hospedou o anúncio
# **host_name** – Contém o nome do usuário que hospedou o anúncio
# **bairro_group** - Contém o nome do bairro onde o anúncio está localizado
# **bairro** - Contém o nome da área onde o anúncio está localizado
# **latitude** - Contém a latitude do local
# **longitude** - Contém a longitude do local
# **room_type** – Contém o tipo de espaço de cada anúncio
# **price** - Contém o preço por noite em dólares listado pelo anfitrião
# **minimo_noites** - Contém o número mínimo de noites que o usuário deve reservar
# **numero_de_reviews** - Contém o número de comentários dados a cada listagem
# **ultima_review** - Contém a data da última revisão dada à listagem
# **reviews_por_mes** - Contém o número de avaliações fornecidas por mês
# **calculado_host_listings_count** - Contém a quantidade de imóveis por host
# **disponibilidade_365** - Contém o número de dias em que o anúncio está disponível para reserva

# ### 1. Faça uma análise exploratória dos dados (EDA), demonstrando as principais características entre as variáveis e apresentando algumas hipóteses de negócio relacionadas. Seja criativo!
# 
# Importando pacotes

# In[208]:


# em geral, prefiro colocar todos os imports em um lugar só
import numpy as np
import pandas as pd
import statistics
import collections
import nltk
import string
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.metrics import mean_absolute_percentage_error, make_scorer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from statistics import mean
from scipy import stats

# import scikit_posthocs as sp

py.init_notebook_mode(connected=True)


# Abrindo o arquivo

# In[209]:


nome_do_arquivo = "C:/Users/elain/Desktop/LIGHTHOUSE/teste_indicium_precificacao.csv"
dados = pd.read_csv(nome_do_arquivo)


# Verificando quantidade de registros e atributos. (16 - 1 é atributo alvo: price)

# In[210]:


dados.shape


# Verificando os tipos de dados

# In[211]:


dados.info()


# Olhando para as estatísticas

# In[212]:


dados.describe()


# Olhando para os 10 últimos registros

# In[213]:


dados.tail(10)


# Olhando para os 10 primeiros registros

# In[214]:


dados.head(10)


# Verificar os NANS

# In[215]:


colunas_nan = dados.columns[dados.isna().any()].tolist()
colunas_nan


# In[216]:


print(dados[colunas_nan])


# In[217]:


linhas_nan = dados[dados.isna().any(axis=1)]
print(linhas_nan)


# Dados ausentes com ISNA

# In[218]:


dados.isna().sum()


# Dados ausentes com ISNULL

# In[219]:


dados.isnull().sum()


# Porcentagem de dados ausentes do atributo NOME com nan

# In[220]:


porcentagem = 16/dados.shape[0]
porcentagem


# Porcentagem de dados ausentes do atributo HOST_NAME com nan

# In[221]:


porcentagem = 21/dados.shape[0]
porcentagem


# 20% de reviews_por_mes e ultima_review são dados ausentes

# In[222]:


porcentagem = 10052/dados.shape[0]
porcentagem


# Verificando se review_por_mes e ultima_review tem distribuição normal. Como vemos nos gráficos a distribuição não é normal. Nesse caso substitui-se os valores ausentes pela mediana. Se fosse uma distribuição normal, os valores ausentes seriam substituido pela média. A média ou a mediana devem ser calculadas sem os valores ausentes.

# In[223]:


fig = go.Figure(data=[go.Histogram(x=dados['reviews_por_mes'])])
fig.update_layout(title='Histograma',
                 xaxis_title='Valores',
                 yaxis_title='Frequencia')
fig.show()


# In[224]:


fig = go.Figure(data=[go.Histogram(x=dados['ultima_review'])])
fig.update_layout(title='Histograma',
                 xaxis_title='Valores',
                 yaxis_title='Frequencia')
fig.show()


# Removendo os valores ausentes do calculo da mediana

# In[225]:


dados2 = dados.dropna()
dados2.head(10)


# Calculando a mediana

# In[226]:


statistics.median(dados2['reviews_por_mes'])


# Substituindo os valores ausentes

# In[227]:


dados['reviews_por_mes'] = dados['reviews_por_mes'].fillna(0.72)
dados.head(10)


# Preenchendo os NANS da data com um valor 0

# In[228]:


dados['ultima_review'] = dados['ultima_review'].fillna('0000-00-00')
dados.head(10)


# Verificando a frequencia das datas

# In[229]:


res = collections.Counter(dados['ultima_review'])
df_res = pd.DataFrame(list(res.items()), columns=['res', 'frequencia'])
df_res.sort_values(by='frequencia', ascending=False)


# Removendo todos os registros em que data é igual a 0000-00-00.

# In[230]:


dados3 = dados[dados['ultima_review'] != '0000-00-00']
dados3.shape


# Agora posso converter essas datas em date time

# In[231]:


datas = pd.to_datetime(dados3['ultima_review'])
datas.tail(10)


# Calculando a mediana das datas que agora são consideradas como série temporais

# In[232]:


datas.median()


# Agora posso substituir todas as datas 0000-00-00 pela data encontrada pela mediana

# In[233]:


dados['ultima_review'] = dados['ultima_review'].replace("0000-00-00", "2019-05-19")
dados.head(10)


# Agora vamos separar mês, dia e ano em novas colunas

# In[234]:


dados[['ano', 'mes', 'dia']] = dados['ultima_review'].str.split('-', expand=True)
dados.head(10)


# Verificando os tipos de dados e atributos com o novo dataframe

# In[235]:


dados.info()


# Dia, mês e ano estão como OBJECT, precisamos converter para inteiro

# In[236]:


colunas = ['ano', 'dia', 'mes']
dados[colunas] = dados[colunas].astype(int)
dados.head(10)


# Também vamos converter a ultima_review para data

# In[237]:


dados['ultima_review'] = pd.to_datetime(dados['ultima_review'])
dados.head(10)


# Conferindo

# In[238]:


dados.info()


# Plotando um gráfico para cada atributo

# In[239]:


for col in dados.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index, y=dados[col], mode='lines', name=col))
    fig.update_layout(title=f'Gráfico de Linhas - {col}',
                      xaxis_title='Índice',
                      yaxis_title='Valor')
    py.iplot(fig)


# Calculando agora as frequencias por ano, dia e mes

# In[240]:


ano_freq = collections.Counter(dados['ano'])
df_ano_freq = pd.DataFrame(list(ano_freq.items()), columns=['ano', 'frequencia'])
df_ano_freq.sort_values(by='ano', ascending=False)


# In[241]:


mes_freq = collections.Counter(dados['mes'])
df_mes_freq = pd.DataFrame(list(mes_freq.items()), columns=['mes', 'frequencia'])
df_mes_freq.sort_values(by='mes', ascending=False)


# In[242]:


dia_freq = collections.Counter(dados['dia'])
df_dia_freq = pd.DataFrame(list(dia_freq.items()), columns=['dia', 'frequencia'])
df_dia_freq.sort_values(by='dia', ascending=False)


# In[ ]:





# Quais são os nomes de anúncios?

# In[243]:


result2 = collections.Counter(dados['nome'])
df = pd.DataFrame(result2.items(), columns=['nome', 'count'])
df_sorted = df.sort_values(by='nome')
df_sorted.to_csv('sorted_result_2.csv', index=False)
df_sorted


# Quais são os bairros?

# In[244]:


bairros_unicos = sorted(dados['bairro'].unique())
df_bairros = pd.DataFrame({'Bairros': bairros_unicos})
print(df_bairros)


# Quais são os tipos de apartamento?

# In[245]:


collections.Counter(dados['room_type'])


# Quantos apartamentos tem cada host?

# In[246]:


result = collections.Counter(dados['host_id'])
df = pd.DataFrame(result.items(), columns=['host_id', 'count'])
df_sorted = df.sort_values(by='count', ascending=False)
df_sorted.to_csv('sorted_result.csv', index=False)
df_sorted


# In[247]:


# 219517861
filtrado2 = dados[dados['host_id'] == 219517861]
pd.DataFrame(filtrado2)


# Quantos anuncios tem cada host em cada grupo de bairros?

# In[248]:


anuncios_por_host = dados.groupby(['host_id', 'bairro_group'])['nome'].size().reset_index(name='num_anuncios')
anuncios_por_host.sort_values(by='num_anuncios', ascending=False)


# Quantos anuncios tem cada host em cada bairros?

# In[249]:


anuncios_por_host = dados.groupby(['host_id', 'bairro'])['nome'].size().reset_index(name='num_anuncios')
anuncios_por_host.sort_values(by='num_anuncios', ascending=False)


# Qual é o número médio "mínimo de noites" por tipo de apartamento?

# In[250]:


dados.groupby('room_type')['minimo_noites'].mean()


# Quais são os tipos de apartamentos com alugueis mais caros e os mais baratos? E em quais bairros estão localizados?

# In[251]:


media_preco_por_tipo = dados.groupby('room_type')['price'].mean()
tipo_mais_caro = media_preco_por_tipo.idxmax()
preco_mais_caro = media_preco_por_tipo.max()
tipo_mais_barato = media_preco_por_tipo.idxmin()
preco_mais_barato = media_preco_por_tipo.min()
bairro_mais_caro = dados.loc[dados['room_type'] == tipo_mais_caro, 'bairro'].unique()
bairro_mais_barato = dados.loc[dados['room_type'] == tipo_mais_barato, 'bairro'].unique()
print(f"O tipo de apartamento mais caro é '{tipo_mais_caro}' com preço médio de ${preco_mais_caro:.2f}, localizado nos bairros: {bairro_mais_caro}.")
print()
print(f"O tipo de apartamento mais barato é '{tipo_mais_barato}' com preço médio de ${preco_mais_barato:.2f}, localizado nos bairros: {bairro_mais_barato}.")
print()


# Quais são os bairros com o maior e menor número de reviews? colocar o 'bairro_group',

# In[252]:


total_reviews_por_bairro = dados.groupby('bairro')['numero_de_reviews'].sum()
bairro_mais_reviews = total_reviews_por_bairro.idxmax()
max_reviews = total_reviews_por_bairro.max()
bairro_menos_reviews = total_reviews_por_bairro.idxmin()
min_reviews = total_reviews_por_bairro.min()
print(f"O bairro com o maior número total de reviews é '{bairro_mais_reviews}' com {max_reviews} reviews.")
print(f"O bairro com o menor número total de reviews é '{bairro_menos_reviews}' com {min_reviews} reviews.")


# Quais são os anuncios com o maior e menor número de reviews? Em que bairro estão?

# In[253]:


maior_reviews = dados.loc[dados['numero_de_reviews'].idxmax()]
print("\nAnuncio com o maior número de reviews:")
print(maior_reviews[['nome', 'bairro', 'bairro_group','numero_de_reviews']])
menor_reviews = dados.loc[dados['numero_de_reviews'].idxmin()]
print("\nAnuncio com o menor número de reviews:")
print(menor_reviews[['nome', 'bairro', 'bairro_group', 'numero_de_reviews']])


# Quais são os anuncios (nome) por bairro? 'bairro_group'

# In[254]:


anuncios_por_bairro = dados.groupby('bairro')['nome'].nunique().reset_index(name='num_tipos_anuncios')
anuncios_por_bairro = anuncios_por_bairro.sort_values(by='num_tipos_anuncios', ascending=False)
print(anuncios_por_bairro)


# Qual é o preço máximo e mínimo?

# In[255]:


preco_maximo = dados['price'].max()
preco_minimo = dados['price'].min()
print("Preço máximo:", preco_maximo)
print("Preço mínimo:", preco_minimo)


# Qual o preço médio do aluguel (por noite) por bairro?

# In[256]:


preco_medio_por_bairro = dados.groupby('bairro')['price'].mean().reset_index()
preco_medio_por_bairro = preco_medio_por_bairro.sort_values(by='price', ascending=False)
print(preco_medio_por_bairro)


# Quanto ganha por noite cada host em média? E por ano?

# In[257]:


ganho_medio_por_noite = dados.groupby('host_id')['price'].mean().reset_index(name='ganho_medio_por_noite')
ganho_medio_por_ano = ganho_medio_por_noite['ganho_medio_por_noite'] * 365

ganho_medio_por_ano = pd.DataFrame({'host_id': ganho_medio_por_noite['host_id'], 'ganho_medio_por_ano': ganho_medio_por_ano})

ganho_medio_por_noite = ganho_medio_por_noite.sort_values(by='ganho_medio_por_noite', ascending=False)
ganho_medio_por_ano = ganho_medio_por_ano.sort_values(by='ganho_medio_por_ano', ascending=False)

print("Ganho médio por noite para cada host:")
print(ganho_medio_por_noite)

print("\nGanho médio por ano para cada host:")
print(ganho_medio_por_ano)


# Qual anuncio tem a disponibilidade mais alta e qual a mais baixa?

# In[258]:


indice_disponibilidade_maxima = dados['disponibilidade_365'].idxmax()
indice_disponibilidade_minima = dados['disponibilidade_365'].idxmin()

anuncio_disponibilidade_maxima = dados.loc[indice_disponibilidade_maxima]
anuncio_disponibilidade_minima = dados.loc[indice_disponibilidade_minima]

print("Anúncio com a disponibilidade mais alta:")
print(anuncio_disponibilidade_maxima)

print("\nAnúncio com a disponibilidade mais baixa:")
print(anuncio_disponibilidade_minima)


# Qual a disponibilidade média por bairro?

# In[259]:


disponibilidade_media_por_bairro = dados.groupby('bairro')['disponibilidade_365'].mean()
df_disponibilidade_media = disponibilidade_media_por_bairro.reset_index()
df_disponibilidade_media.rename(columns={'disponibilidade_365': 'disponibilidade_media'}, inplace=True)
df_disponibilidade_media = df_disponibilidade_media.sort_values(by='disponibilidade_media', ascending=False)
print(df_disponibilidade_media)


# Qual o preço médio do aluguel por grupo de bairro?

# In[260]:


preco_medio_por_bairro = dados.groupby('bairro_group')['price'].mean().reset_index()
preco_medio_por_bairro = preco_medio_por_bairro.sort_values(by='price', ascending=False)
print(round(preco_medio_por_bairro))


# Qual a disponibilidade média (dada por dia) por grupo de bairro?

# In[261]:


disponibilidade_media_por_bairro_group = dados.groupby('bairro_group')['disponibilidade_365'].mean().reset_index(name='disponibilidade_media')
disponibilidade_media_por_bairro_group_ordenado = disponibilidade_media_por_bairro_group.sort_values(by='disponibilidade_media', ascending=False)
print("Disponibilidade média por bairro_group:")
print(round(disponibilidade_media_por_bairro_group_ordenado))


# In[262]:


merged_df = pd.merge(disponibilidade_media_por_bairro_group, preco_medio_por_bairro, on='bairro_group')
print(merged_df)


# Qual é a média de preço para cada tipo de apartamento?

# In[263]:


preco_media_por_tipo = dados.groupby('room_type')['price'].mean().reset_index(name='media_preco')
print("Preço médio por tipo de apartamento:")
print(round(preco_media_por_tipo))


# Qual a disponibilidade média (dada por dia) por tipo de apartamento?

# In[264]:


disponibilidade_media_por_tipo = dados.groupby('room_type')['disponibilidade_365'].mean().reset_index(name='media_disponibilidade')
print("Disponibilidade média por tipo de apartamento:")
print(round(disponibilidade_media_por_tipo))


# In[265]:


merged_df = pd.merge(preco_media_por_tipo, disponibilidade_media_por_tipo, on='room_type')
print(merged_df)


# Um apartamento com alta disponibilidade (por dia) rende mais que um apartamento com baixa disponibilidade? Vamos considerar a média como um ponto de referencia. 

# In[266]:


# Calcular a média da disponibilidade
media_disponibilidade = dados['disponibilidade_365'].mean()

# Dividir os dados em grupos com alta e baixa disponibilidade
alta_disponibilidade = dados[dados['disponibilidade_365'] > media_disponibilidade]
baixa_disponibilidade = dados[dados['disponibilidade_365'] < media_disponibilidade]

# Calcular a média do preço por noite para cada grupo
media_preco_alta_disponibilidade = alta_disponibilidade['price'].mean()
media_preco_baixa_disponibilidade = baixa_disponibilidade['price'].mean()

# Comparar as médias de preço
if media_preco_alta_disponibilidade > media_preco_baixa_disponibilidade:
    print("Apartamentos com alta disponibilidade rendem mais.")
elif media_preco_alta_disponibilidade < media_preco_baixa_disponibilidade:
    print("Apartamentos com baixa disponibilidade rendem mais.")
else:
    print("Não há diferença significativa no rendimento entre apartamentos com alta e baixa disponibilidade.")


# Qual a média de dias de disponibilidade?

# In[267]:


media_disponibilidade = dados['disponibilidade_365'].mean()
print("A média de dias é:", media_disponibilidade)


# E para cada anuncio?

# In[268]:


media_disponibilidade_por_anuncio = dados.groupby('nome')['disponibilidade_365'].mean().reset_index()
media_disponibilidade_por_anuncio = media_disponibilidade_por_anuncio.sort_values(by='disponibilidade_365', ascending=False)
print(media_disponibilidade_por_anuncio)


# Selecionando os 10 bairros mais caros

# In[269]:


preco_medio_por_bairro = dados.groupby('bairro')['price'].mean().reset_index()
top_10_bairros_caros = preco_medio_por_bairro.sort_values(by='price', ascending=False).head(10)
print(top_10_bairros_caros)


# Selecionando os 10 anuncios mais caros

# In[270]:


preco_medio_por_bairro = dados.groupby('nome')['price'].mean().reset_index()
top_10_bairros_caros = preco_medio_por_bairro.sort_values(by='price', ascending=False).head(10)
print(top_10_bairros_caros)


# Selecionando os 10 anuncios (atributo nome) mais caros por group bairro

# In[271]:


top10_por_bairro = dados.groupby('bairro').apply(lambda x: x.nlargest(10, 'price'))
top10_por_bairro = top10_por_bairro.reset_index(drop=True)
top10_nomes_por_bairro = top10_por_bairro[['nome', 'bairro', 'price']]
print(top10_nomes_por_bairro)


# Selecionando os 10 anuncios (nome) mais caros por group bairro

# In[272]:


top10_por_grupo_bairro = dados.groupby('bairro_group').apply(lambda x: x.nlargest(10, 'price'))
top10_por_grupo_bairro = top10_por_grupo_bairro.reset_index(drop=True)
top10_nomes_por_grupo_bairro = top10_por_grupo_bairro[['nome', 'bairro_group', 'price']]
print(top10_nomes_por_grupo_bairro)


# Quais atributos tem dependências com outros? 
# 
# Algumas possibilidades: 
# 
# a) price e room_type 
# 
# b) price e minimo_noites
# 
# c) price e disponibilidade_365
# 
# d) price e bairro 
# 
# e) price e bairro_group
# 
# f) numero_de_reviews e reviews_por_mes
# 
# Calcular: covariância, correlação e determinação.
# 
# CORRELAÇÃO
# 
# O valor 1.0 indica alta correlação e ele aparece na diagonal pois é onde os atributos cruzam com eles mesmos. Na correlação, a variabilidade de uma variável pode ser explicada pela variação de outra variável. Neste case, podemos concluir que a correlação entre os atributos selecionados são baixas, isto é, há pouca relação entre eles.

# In[273]:


subset = dados[['price', 'minimo_noites', 'disponibilidade_365']]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(subset)
normalized_df = pd.DataFrame(normalized_data, columns=subset.columns)
correlacao = normalized_df.corr()
print(correlacao)


# COVARIÂNCIA
# 
# Descobrimos com a covariância se existe alguma relação entre as variáveis. 
# 
# Se o resultado for maior que zero então significa que existe uma dependência entre os atributos. Neste caso, quanto maior a variável1 maior também a variável2.
# 
# Se o resultado for menor que zero, também há dependência. Daí indica que quanto menor a variável 1, menor a variável 2. 
# 
# Se o resultado for igual a zero, então as variáveis são independentes. 
# 
# Em nosso case a covariância é positiva, indicando que quanto maior a variável1 maior também a variável2.

# In[274]:


covariancia = normalized_df.cov()
print(covariancia)


# COEFICIENTE DE DETERMINAÇÃO
# 
# Se o resultado for ~1, mais a primeira variável explica a segunda
# 
# Se o resultado for ~0, menos a primeira variável (atributo) explica a segunda, isto é, uma porcentagem da segunda variável pode ser explicada pela primeira. 
# 
# Os valores obtidos para este dataset são mais próximos de zero do que de um, portanto, um atributo não explica muito bem os outros atributos.

# In[275]:


determinacao = correlacao.pow(2)
print(determinacao)


# b) price e minimo_noites
# 
# c) price e disponibilidade_365
# 
# a) price e room_type 
# 
# d) price e bairro 
# 
# e) price e bairro_group
# 
# f) numero_de_reviews e reviews_por_mes

# In[276]:


# Plotar o gráfico
fig = px.scatter(dados, x='price', y='minimo_noites', color='room_type', size='disponibilidade_365',
                 labels={'price': 'Preço', 'minimo_noites': 'Mínimo de Noites', 'disponibilidade_365': 'Disponibilidade 365'},
                 title='')

# Exibir o gráfico
fig.show()

# X: preço 
# Y: número mínimo de noites
# O tamanho dos pontos representa a disponibilidade_365
# A cor dos pontos representa o room_type


# In[277]:


# Plotar o gráfico
fig = px.scatter(dados, x='price', y='minimo_noites', color='bairro_group', size='disponibilidade_365',
                 labels={'price': 'Preço', 'minimo_noites': 'Mínimo de Noites', 'disponibilidade_365': 'Disponibilidade 365'},
                 title='')

# Exibir o gráfico
fig.show()

# X: preço 
# Y: número mínimo de noites
# O tamanho dos pontos representa a disponibilidade_365
# A cor dos pontos representa o grupo de bairros


# Existe algum padrão no texto do nome do local para lugares de mais alto valor?
# Podemos usar processamento de linguagem natural pra isso.

# In[278]:


# Baixar as stopwords (se necessário)
nltk.download('stopwords')
nltk.download('punkt')


# In[279]:


# Calcular o preço médio por bairro
preco_medio_por_bairro = dados.groupby('bairro')['price'].mean().reset_index()

# Ordenar os bairros com base no preço médio em ordem decrescente e selecionar os 10 primeiros
top_10_bairros_caros = preco_medio_por_bairro.sort_values(by='price', ascending=False).head(10)

# Combinar o DataFrame original com os 10 bairros mais caros
dados_top_10_bairros_caros = pd.merge(dados, top_10_bairros_caros, on='bairro')
dados_top_10_bairros_caros

# Selecionar os nomes dos locais (bairros)
nomes_locais = dados_top_10_bairros_caros['nome'].tolist()
nomes_locais


# In[280]:


# Tokenização e limpeza dos nomes dos locais
tokens_limpos = []
stop_words = set(stopwords.words('english'))
for nome_local in nomes_locais:
    if isinstance(nome_local, str):  # Verificar se o valor não é NaN
        tokens = word_tokenize(nome_local.lower())  # Tokenização e conversão para minúsculas
        tokens_limpos.extend([token for token in tokens if token not in stop_words and token.isalpha()])

# Calcular a frequência das palavras
frequencia_palavras = pd.Series(tokens_limpos).value_counts()

# Visualizar as palavras mais frequentes
print(frequencia_palavras.head(20))


# Aparentemente, nos 10 bairros mais caros, essas são as palavras envolvidas que podem indicar o porque do valor ser mais alto nesses bairros comparados à outros. LOFT definitivamente diz muito sobre o imóvel e geralmente esse tipo costuma mesmo ter um alto valor.

# Explique como você faria a previsão do preço a partir dos dados.
# - Quais variáveis e/ou suas transformações você utilizou e por quê?
# - Qual tipo de problema estamos resolvendo (regressão, classificação)?
# - Qual modelo melhor se aproxima dos dados e quais seus prós e contras?
# - Qual medida de performance do modelo foi escolhida e por quê?
# 
# Para responder a estas perguntas teremos de selecionar atributos que realmente são relevantes para a predição. Este é um caso de **REGRESSÃO**, isto é, precisamos predizer um valor contínuo. Nem todos os atributos de entrada servirão para este propósito. Analisando cada atributo:
# 
# + id – não acrescenta nenhum tipo de informação relevante 
# + nome - contém informações uteis, mas são todos caracteres! 
# + host_id - não acrescenta nenhum tipo de informação relevante 
# + host_name – não acrescenta nenhum tipo de informação relevante 
# + bairro_group - pode ser útil, mas é string! teria que transformar 
# + bairro - pode ser útil, mas é string! teria que transformar 
# + latitude - útil já que é uma forma de identificar a localização do bairro sem ser por uma string, 
# + longitude - útil já que é uma forma de identificar a localização do bairro sem ser por uma string, 
# + room_type – útil pois o tipo de quarto pode aumentar ou diminuir o valor do aluguel 
# + price - esse é o atributo alvo! 
# + minimo_noites - útil para estimar um valor mínimo 
# + numero_de_reviews - útil - significa que muitas pessoas alugaram aquele espaço 
# + ultima_review - pode vir a ser útil, mas sendo data, é necessário fazer uma transformação, mas talvez isso não contribua muito
# + reviews_por_mes - útil, mas sendo strings se torna mais complexo para predição de números. 
# + calculado_host_listings_count - Contém a quantidade de anuncios total por host
# + disponibilidade_365 - talvez seja importante
# 
# Sem conversão de tipos podemos usar id, host_id, latitude, longitue, minimo_noites, numero_de_reviews, reviews_por_mes, disponibilidade_365. Os atributos room_type, bairro_group e bairro tem que ser convertidos. 

# In[281]:


mapeamento_room_type = {'Entire home/apt': 1, 'Private room': 2, 'Shared room': 3}
dados['room_type_numerico'] = dados['room_type'].map(mapeamento_room_type)
print(dados[['room_type', 'room_type_numerico']].head(20))


# In[282]:


bairro_group_mapping = {'Bronx': 1, 'Brooklyn': 2, 'Manhattan': 3, 'Queens': 4, 'Staten Island': 5}
dados['bairro_group_numerico'] = dados['bairro_group'].map(bairro_group_mapping)
print(dados[['bairro_group', 'bairro_group_numerico']].head(20))


# In[283]:


bairros_unicos = dados['bairro'].unique()
mapeamento_bairros = {bairro: i+1 for i, bairro in enumerate(bairros_unicos)}
dados['bairro_numerico'] = dados['bairro'].map(mapeamento_bairros)
print(dados[['bairro', 'bairro_numerico']].head(20))


# In[284]:


dados.head(10)


# Vamos criar um novo dataframe que conterá apenas os atributos de interesse

# In[285]:


dados.columns


# In[286]:


new_data = dados[['host_id', 'latitude', 'longitude', 'minimo_noites',
                        'numero_de_reviews', 'reviews_por_mes', 'disponibilidade_365',
                        'room_type_numerico', 'bairro_group_numerico', 'bairro_numerico',
                  'price']]
print( new_data.head(10))


# In[287]:


X = new_data.drop(columns=['price']) # atributos de entrada
y = new_data['price'] # atributo alvo - número

# holdout: 30% teste, 70% treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# normalize
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html

# padronização dos valores
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# https://scikit-learn.org/stable/modules/linear_model.html
modelos = {
    'Regressão Linear': LinearRegression(),    
    'Regressão Ridge': Ridge(alpha=0.5),
    'Regressão Lasso': Lasso(alpha=0.5),
    'Árvore de Decisão': DecisionTreeRegressor(random_state=42)
    
    # Esses modelos estavam travando meu computador
    # 'Random Forest': RandomForestRegressor(random_state=42),
    # 'SVR': SVR(kernel='rbf'),    
}


# In[288]:


resultados = pd.DataFrame(columns=['Nome', 'MAE', 'MSE', 'RMSE', 'R2'])
for nome, modelo in modelos.items():    
    modelo.fit(X_train_scaled, y_train) # fit = treinar
    y_pred = modelo.predict(X_test_scaled) # predict = predizer
    
    #modelo.fit(X_train, y_train)
    #y_pred = modelo.predict(X_test)
    
    # Calcular as métricas
    # mpe = mean_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)    
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)    
    r2 = r2_score(y_test, y_pred)  
    
    temp_df = pd.DataFrame({'Nome': [nome],
                            'MAE': [mae],
                            'MSE': [mse],
                            'RMSE': [rmse],
                            'MAPE': [mape],
                            'R2': [r2]})   
    
    resultados = pd.concat([resultados, temp_df], ignore_index=True)    
    
    predicted = np.round(y_pred,2)    
    saidas = pd.DataFrame([y_test, predicted]).T
    saidas.columns = ["Real", "Predito"]    
    
    # Exibir os resultados
    # print(f'{nome}:')    
    # print(f'MAE = {mae}')
    # print(f'MSE = {mse}')
    # print(f'RMSE = {rmse}')
    # print(f'R² = {r2}')
    # print("Preços Preditos:", predicted)
    # print()
    
print(resultados)
print(saidas)


# Usando validação cruzada de 10 folds

# In[289]:


# Definir as métricas a serem usadas para avaliação
scorer_mse = make_scorer(mean_squared_error)
scorer_mape = make_scorer(lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
scorer_mpe = make_scorer(lambda y_true, y_pred: np.mean((y_true - y_pred) / y_true) * 100)
scorer_rmse = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))
scorer_r2 = make_scorer(r2_score)
scorer_mae = make_scorer(mean_absolute_error)

# Inicializar os modelos de regressão
modelos = {
    'Regressão Linear': LinearRegression(),
    'Árvore de Decisão': DecisionTreeRegressor(random_state=42),
    'Regressão Ridge': Ridge(alpha=0.5),
    'Regressão Lasso': Lasso(alpha=0.5)
    # 'SVR': SVR(kernel='rbf'),
    #'Random Forest': RandomForestRegressor(random_state=42),
}


# In[290]:


# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Avaliar cada modelo utilizando validação cruzada de 10 folds
for nome, modelo in modelos.items():
    # Calcular os scores de validação cruzada para o modelo
    scores_mse = cross_val_score(modelo, X, y, cv=10, scoring=scorer_mse)
    scores_mape = cross_val_score(modelo, X, y, cv=10, scoring=scorer_mape)
    scores_mpe = cross_val_score(modelo, X, y, cv=10, scoring=scorer_mpe)
    scores_rmse = cross_val_score(modelo, X, y, cv=10, scoring=scorer_rmse)
    scores_r2 = cross_val_score(modelo, X, y, cv=10, scoring=scorer_r2)
    scores_mae = cross_val_score(modelo, X, y, cv=10, scoring=scorer_mae)
    
    # Obter as previsões durante a validação cruzada
    predicted = cross_val_predict(modelo, X, y, cv=10)
    predicted = np.round(predicted,2)
    
    # Exibir os resultados
    print(f'{nome}:')    
    print(f'Média do MSE: {np.mean(scores_mse)}')
    print(f'Média do MAPE: {np.mean(scores_mape)}')
    print(f'Média do MPE: {np.mean(scores_mpe)}')
    print(f'Média do RMSE: {np.mean(scores_rmse)}')
    print(f'Média do R2: {np.mean(scores_r2)}')
    print(f'Média do MAE: {np.mean(scores_mae)}')
    print(f'Preços Preditos: {predicted}')    
    print()


# O modelo de regressão logística teve um melhor desempenho que o modelo de árvore de decisão. Seria necessário fazer um tunning mais profundo de parametros, para se chegar a resultados de desempenho melhores. Eu escolhi esses modelos por eles serão os básico e padrões. Os outros modelos tentei executar, mas meu computador é muito velho e não deu conta.
# 
# A R2 é a melhor medida pois tem um valor mais baixo que o dos outros modelos, já que para modelos de regressão, quanto menor o valor do resultado, melhor. De qualquer forma, o modelo não conseguiu predizer alguns valores. Infelizmente ainda não sei muito sobre regressão e, portanto, não consigo dar muitos insights a respeito.
