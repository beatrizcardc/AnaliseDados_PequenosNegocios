import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind, f_oneway

# Configuração da página
st.set_page_config(page_title="Datalyze - Análise Inteligente de Negócios", layout="wide")

# Título do App
st.title("📊 Datalyze - Análise Inteligente de Negócios")
st.write("Bem-vindo! Aqui você pode carregar seus dados e aplicar técnicas de análise para obter insights valiosos.")

# Explicação das técnicas
st.sidebar.subheader("📌 Sobre as Análises Disponíveis")
st.sidebar.write("**Previsão de Vendas:** Usa regressão linear para estimar vendas futuras com base em fatores como dia da semana, horário e temperatura.")
st.sidebar.write("**Clusterização de Clientes:** Identifica grupos de clientes com padrões de compra semelhantes para campanhas personalizadas.")
st.sidebar.write("**Testes Estatísticos:** Compara diferentes grupos de vendas para entender se mudanças no negócio tiveram impacto significativo.")

# Função para carregar e exibir dados
def carregar_dados():
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV/XLS", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            xls = pd.ExcelFile(uploaded_file)
            planilhas = xls.sheet_names
            sheet_selecionada = st.sidebar.selectbox("Escolha a planilha:", planilhas)
            df = pd.read_excel(xls, sheet_name=sheet_selecionada)
        
        # Verifica se a coluna de data existe e adiciona o filtro
        # Verifica se a coluna de data existe, mas não adiciona `date_input` aqui
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'])
        else:
            st.warning("A coluna 'data' não foi encontrada no arquivo. Verifique se selecionou a planilha correta.")
        
        st.session_state['df'] = df
        return df
    return None

# Função para calcular o Top 10 produtos mais vendidos
def top_10_produtos(df):
    if 'Produto' in df.columns and 'Vendas' in df.columns:
        top_produtos = df.groupby("Produto")["Vendas"].sum().nlargest(10).index
        return df[df["Produto"].isin(top_produtos)]
    return df

# Função de previsão de vendas
def prever_vendas(df):
    if {'dia_semana', 'horario', 'temperatura', 'vendas'}.issubset(df.columns):
        X = df[['dia_semana', 'horario', 'temperatura']]
        y = df['vendas']
        modelo = LinearRegression().fit(X, y)
        df['previsao_vendas'] = modelo.predict(X)
        return df
    else:
        st.warning("O arquivo precisa conter as colunas: dia_semana, horario, temperatura, vendas. Por favor, verifique se selecionou a planilha correta. Para a análise de previsão de vendas, selecione a planilha de 'Vendas'.")
        return None

# Função de clusterização
def clusterizar_clientes(df):
    if {'idade', 'frequencia_compra', 'gasto_medio'}.issubset(df.columns):
        kmeans = KMeans(n_clusters=3, random_state=42).fit(df[['idade', 'frequencia_compra', 'gasto_medio']])
        df['cluster'] = kmeans.labels_
        return df
    else:
        st.warning("O arquivo precisa conter as colunas: idade, frequencia_compra, gasto_medio. Por favor, verifique se selecionou a planilha correta. Para a análise de clusterização, selecione a planilha de 'Clientes'.")
        return None

# Função de testes estatísticos
def testes_estatisticos(df):
    if {'grupo', 'vendas'}.issubset(df.columns):
        grupos = df.groupby('grupo')['vendas'].apply(list)
        explicacao = "O Teste T é usado para comparar a média de dois grupos distintos e verificar se há diferença estatisticamente significativa entre eles. Se o p-valor for menor que 0.05, rejeitamos a hipótese nula, indicando que há uma diferença significativa. Caso contrário, não há evidências suficientes para afirmar que os grupos são diferentes."
        if len(grupos) == 2:
            stat, p = ttest_ind(grupos.iloc[0], grupos.iloc[1])
            return "Teste T", p, explicacao
        elif len(grupos) > 2:
            stat, p = f_oneway(*grupos)
            explicacao = "A Análise de Variância (ANOVA) é utilizada para comparar a média de três ou mais grupos e verificar se pelo menos um deles é significativamente diferente dos outros. Se o p-valor for menor que 0.05, há evidências de que pelo menos um grupo é diferente."
            return "ANOVA", p, explicacao
        else:
            return None, None, ""
    else:
        return None, None, ""

# Sidebar
# Filtro de período para todas as análises
if 'df' in st.session_state and st.session_state['df'] is not None:
    df = st.session_state['df']
    
    #Verificar se 'data' existe antes de aplicar a conversão
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'])
        data_min, data_max = df['data'].min(), df['data'].max()
        data_inicio, data_fim = st.sidebar.date_input("Selecione o período:", [data_min, data_max], data_min, data_max)
        df = df[(df['data'] >= pd.Timestamp(data_inicio)) & (df['data'] <= pd.Timestamp(data_fim))]
    else:
        st.warning("A coluna 'data' não foi encontrada no arquivo. Verifique se selecionou a planilha correta.")

st.sidebar.title("📂 Opções de Análise")


# Carregar os dados
df = carregar_dados()
if df is not None:
    df = top_10_produtos(df)

    st.write("### 📋 Dados Carregados (Top 10 Produtos Mais Vendidos)")
    st.dataframe(df.head())

    analise_selecionada = st.sidebar.selectbox(
    "Escolha uma análise", 
    ["Previsão de Vendas", "Clusterização de Clientes", "Testes Estatísticos"], 
    key="analise_selecao"
)


#if df is not None:
    #st.write("### 📋 Dados Carregados")
    #st.dataframe(df.head())

    if analise_selecionada == "Previsão de Vendas":
        # Adiciona a opção para o usuário escolher a variável para visualização do gráfico
        variavel_grafico = st.sidebar.selectbox("Escolha a variável para visualizar a previsão:", ["horario", "dia_semana", "temperatura"])
        df = prever_vendas(df)
        if df is not None:
            st.write("### 📈 Previsão de Vendas")
            st.write(f"### 📈 Previsão de Vendas em função de {variavel_grafico.capitalize()}")
            
            if variavel_grafico == 'dia_semana':
                dias_semana = {1: 'Segunda', 2: 'Terça', 3: 'Quarta', 4: 'Quinta', 5: 'Sexta', 6: 'Sábado', 7: 'Domingo'}
                df['dia_semana'] = df['dia_semana'].map(dias_semana)
            
            df_plot = df[[variavel_grafico, 'vendas', 'previsao_vendas']].groupby(variavel_grafico).mean()
            st.write(f"### 📈 Previsão de Vendas vs. Vendas Reais em função de {variavel_grafico.capitalize()}")
            st.line_chart(df_plot)

    if analise_selecionada == "Clusterização de Clientes":
        df = clusterizar_clientes(df)
        if df is not None:
            st.write("### 👥 Segmentação de Clientes")
            fig, ax = plt.subplots()
            for cluster in df['cluster'].unique():
                cluster_data = df[df['cluster'] == cluster]
                ax.scatter(cluster_data['idade'], cluster_data['gasto_medio'], label=f'Cluster {cluster}')
            ax.set_xlabel('Idade')
            ax.set_ylabel('Gasto Médio')
            ax.legend()
            st.pyplot(fig)
    
    elif analise_selecionada == "Testes Estatísticos":
        teste, p, explicacao = testes_estatisticos(df)
        if teste:
            st.write(f"### 📊 Resultado do {teste}")
            st.write(f"p-valor: {p:.4f}")
            st.write(f"📌 **Explicação:** {explicacao}")
            if p < 0.05:
                st.success("Diferença estatisticamente significativa encontrada! Isso indica que os grupos analisados possuem médias diferentes com uma confiança maior que 95%.")
            else:
                st.info("Nenhuma diferença significativa encontrada. Isso sugere que os grupos analisados têm médias semelhantes.")
    
    st.sidebar.button("🗑️ Limpar Dados", on_click=lambda: st.session_state.pop('df', None))


# Rodapé
st.markdown("---")
st.markdown("**📧 Contato:** Beatriz Cardoso Cunha | Email: beacarcun@gmail.com | LinkedIn: https://www.linkedin.com/in/beatriz-cardoso-cunha/")
