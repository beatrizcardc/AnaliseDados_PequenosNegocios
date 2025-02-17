import streamlit as st
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

# Sidebar para carregamento de dados
st.sidebar.title("📂 Carregar Dados")
uploaded_file = st.sidebar.file_uploader("Carregar arquivo XLSX", type=["xlsx"])

df_vendas, df_clientes, df_testes = None, None, None

if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    df_vendas = pd.read_excel(xls, sheet_name="Vendas")
    df_clientes = pd.read_excel(xls, sheet_name="Clientes")
    df_testes = pd.read_excel(xls, sheet_name="Testes")
    
    # Garantir que a coluna de data está no formato correto
    df_vendas["Data"] = pd.to_datetime(df_vendas["Data"]).dt.date
    
    # Filtro de período
    data_min, data_max = df_vendas["Data"].min(), df_vendas["Data"].max()
    data_inicio, data_fim = st.sidebar.date_input("Selecione o período:", [data_min, data_max], data_min, data_max, key="data_selecao")
    
    df_vendas = df_vendas[(df_vendas["Data"] >= data_inicio) & (df_vendas["Data"] <= data_fim)]
    
    st.sidebar.success("Dados carregados com sucesso!")

    # Seção 1: Top 10 Produtos Mais Vendidos
    st.header("🏆 Top 10 Produtos Mais Vendidos")
    top_10_produtos = df_vendas.groupby("Produto")["Vendas"].sum().nlargest(10)
    fig, ax = plt.subplots()
    top_10_produtos.plot(kind="bar", ax=ax, color="royalblue")
    ax.set_title("Top 10 Produtos Mais Vendidos")
    ax.set_xlabel("Produto")
    ax.set_ylabel("Quantidade Vendida")
    st.pyplot(fig)

    # Seção 2: Perfil dos Clientes por Produto
    st.header("👥 Perfil dos Clientes por Produto")
    produto_selecionado = st.selectbox("Selecione um produto para análise de perfil de clientes:", df_vendas["Produto"].unique())
    df_clientes_produto = df_vendas[df_vendas["Produto"] == produto_selecionado].merge(df_clientes, on="Nome do Cliente", how="left")
    
    if not df_clientes_produto.empty:
        st.write("**Idade Média:**", round(df_clientes_produto["Idade"].mean(), 1))
        st.write("**Ticket Médio:** R$", round(df_clientes_produto["Gasto Médio"].mean(), 2))
        st.write("**Frequência de Compra Média:**", round(df_clientes_produto["Frequência de Compra"].mean(), 1))
        
        # Gráfico de distribuição de idade
        fig, ax = plt.subplots()
        df_clientes_produto["Idade"].hist(bins=10, ax=ax, color="teal")
        ax.set_title("Distribuição de Idade dos Compradores")
        ax.set_xlabel("Idade")
        ax.set_ylabel("Quantidade")
        st.pyplot(fig)
    else:
        st.warning("Não há dados suficientes para este produto.")

    # Seção 3: Previsão de Vendas
    st.header("📈 Previsão de Vendas")
    df_vendas["Dia da Semana"] = df_vendas["Dia da Semana"].astype(int)
    X = df_vendas[["Dia da Semana", "Temperatura"]]
    y = df_vendas["Vendas"]
    modelo = LinearRegression().fit(X, y)
    df_vendas["Previsão"] = modelo.predict(X)
    
    fig, ax = plt.subplots()
    df_vendas.groupby("Dia da Semana")["Previsão"].mean().plot(kind="line", marker="o", ax=ax, color="red")
    ax.set_title("Previsão de Vendas por Dia da Semana")
    ax.set_xlabel("Dia da Semana")
    ax.set_ylabel("Vendas Previstas")
    st.pyplot(fig)

    # Seção 4: Testes Estatísticos
    st.header("📊 Testes Estatísticos")
    grupos = df_testes.groupby("Grupo")["Vendas"].apply(list)
    if len(grupos) == 2:
        stat, p = ttest_ind(grupos.iloc[0], grupos.iloc[1])
        st.write(f"**Teste T:** p-valor = {p:.4f}")
    elif len(grupos) > 2:
        stat, p = f_oneway(*grupos)
        st.write(f"**ANOVA:** p-valor = {p:.4f}")
    else:
        st.warning("Não há grupos suficientes para realizar testes estatísticos.")

# Rodapé
st.markdown("---")
st.markdown("**📧 Contato:** Beatriz Cardoso Cunha | Email: beacarcun@gmail.com | LinkedIn: [Perfil LinkedIn](https://www.linkedin.com/in/beatriz-cardoso-cunha/)")

