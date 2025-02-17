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
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'])
            data_min, data_max = df['data'].min(), df['data'].max()
            st.sidebar.subheader("📆 Filtro de Período")
            data_inicio, data_fim = st.sidebar.date_input("Selecione o período:", [data_min, data_max], data_min, data_max)
            df = df[(df['data'] >= pd.Timestamp(data_inicio)) & (df['data'] <= pd.Timestamp(data_fim))]
        
        st.session_state['df'] = df
        return df
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
        if len(grupos) == 2:
            stat, p = ttest_ind(grupos.iloc[0], grupos.iloc[1])
            return "Teste T", p
        elif len(grupos) > 2:
            stat, p = f_oneway(*grupos)
            return "ANOVA", p
        else:
            return None, None
    else:
        st.warning("O arquivo precisa conter as colunas: grupo, vendas. Por favor, verifique se selecionou a planilha correta. Para os testes estatísticos, selecione a planilha de 'Testes'.")
        return None, None

# Sidebar
st.sidebar.title("📂 Opções de Análise")
analise_selecionada = st.sidebar.selectbox("Escolha uma análise", ["Previsão de Vendas", "Clusterização de Clientes", "Testes Estatísticos"])
df = carregar_dados()

if df is not None:
    st.write("### 📋 Dados Carregados")
    st.dataframe(df.head())

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
        teste, p = testes_estatisticos(df)
        if teste:
            st.write(f"### 📊 Resultado do {teste}")
            st.write(f"p-valor: {p:.4f}")
            if p < 0.05:
                st.success("Diferença estatisticamente significativa encontrada!")
            else:
                st.info("Nenhuma diferença significativa encontrada.")

    st.sidebar.button("🗑️ Limpar Dados", on_click=lambda: st.session_state.pop('df', None))


# Rodapé
st.markdown("---")
st.markdown("**📧 Contato:** Beatriz Cardoso Cunha | Email: beacarcun@gmail.com | LinkedIn: https://www.linkedin.com/in/beatriz-cardoso-cunha/")
