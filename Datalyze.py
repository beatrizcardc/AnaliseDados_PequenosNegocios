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

# Função para calcular o Top 10 produtos mais vendidos
def top_10_produtos(df):
    if 'Produto' in df.columns and 'Vendas' in df.columns:
        top_produtos = df.groupby("Produto")["Vendas"].sum().nlargest(10).index
        return df[df["Produto"].isin(top_produtos)]
    return df

# Carregar os dados
df = carregar_dados()

if df is not None:
    # Aplicar filtro de Top 10 produtos mais vendidos
    df = top_10_produtos(df)

    st.write("### 📋 Dados Carregados (Top 10 Produtos Mais Vendidos)")
    st.dataframe(df.head())

    # Criar um seletor para escolher um produto do Top 10
    if 'Produto' in df.columns:
        produtos_disponiveis = df["Produto"].unique()
        produto_selecionado = st.sidebar.selectbox("Selecione um produto do Top 10:", produtos_disponiveis)

        # Filtrar os dados para o produto escolhido
        df_produto = df[df["Produto"] == produto_selecionado]

        # Criar um gráfico de vendas ao longo do tempo
        st.write(f"### 📈 Vendas do Produto: {produto_selecionado}")
        fig, ax = plt.subplots()
        ax.plot(df_produto["data"], df_produto["Vendas"], marker="o", linestyle="-")
        ax.set_xlabel("Data")
        ax.set_ylabel("Vendas")
        ax.set_title(f"Vendas do Produto: {produto_selecionado}")
        st.pyplot(fig)

# Rodapé
st.markdown("---")
st.markdown("**📧 Contato:** Beatriz Cardoso Cunha | Email: beacarcun@gmail.com | LinkedIn: https://www.linkedin.com/in/beatriz-cardoso-cunha/")

