import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind, f_oneway

# Configuração da página
st.set_page_config(page_title="Datalyze - Análise Inteligente de Negócios", layout="wide")

# Sidebar para carregamento de dados
st.sidebar.title("📂 Carregar Dados")
uploaded_file = st.sidebar.file_uploader("Carregar arquivo XLSX", type=["xlsx"])

# Sidebar para seleção de análise
st.sidebar.title("🔍 Opções de Análise")
analise_selecionada = st.sidebar.selectbox("Escolha a análise:", ["Top 10 Produtos", "Perfil dos Clientes", "Previsão de Vendas", "Testes Estatísticos", "Clusterização de Clientes"])

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

    if analise_selecionada == "Top 10 Produtos":
        st.header("🏆 Top 10 Produtos Mais Vendidos")
        top_10_produtos = df_vendas.groupby("Produto")["Vendas"].sum().nlargest(10)
        fig, ax = plt.subplots()
        top_10_produtos.plot(kind="bar", ax=ax, color="royalblue")
        ax.set_title("Top 10 Produtos Mais Vendidos")
        ax.set_xlabel("Produto")
        ax.set_ylabel("Quantidade Vendida")
        st.pyplot(fig)

    elif analise_selecionada == "Perfil dos Clientes":
        st.header("👥 Perfil dos Clientes por Produto")
        produto_selecionado = st.sidebar.selectbox("Selecione um produto:", df_vendas["Produto"].unique())
        
        if "Nome do Cliente" in df_vendas.columns and "Nome do Cliente" in df_clientes.columns:
            df_clientes_produto = df_vendas[df_vendas["Produto"] == produto_selecionado].merge(df_clientes, on="Nome do Cliente", how="left")
            
            if not df_clientes_produto.empty:
                st.write("**Idade Média:**", round(df_clientes_produto["Idade"].mean(), 1))
                st.write("**Ticket Médio:** R$", round(df_clientes_produto["Gasto Médio"].mean(), 2))
                st.write("**Frequência de Compra Média:**", round(df_clientes_produto["Frequência de Compra"].mean(), 1))
                
                fig, ax = plt.subplots()
                df_clientes_produto["Idade"].hist(bins=10, ax=ax, color="teal")
                ax.set_title("Distribuição de Idade dos Compradores")
                ax.set_xlabel("Idade")
                ax.set_ylabel("Quantidade")
                st.pyplot(fig)
            else:
                st.warning("Não há dados suficientes para este produto.")
        else:
            st.warning("A coluna 'Nome do Cliente' não foi encontrada em uma das planilhas.")
    
    elif analise_selecionada == "Previsão de Vendas":
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
    
    elif analise_selecionada == "Clusterização de Clientes":
        st.header("🔍 Clusterização de Clientes")
        if "Idade" in df_clientes.columns and "Gasto Médio" in df_clientes.columns:
            kmeans = KMeans(n_clusters=3, random_state=42).fit(df_clientes[["Idade", "Gasto Médio"]])
            df_clientes["Cluster"] = kmeans.labels_
            
            fig, ax = plt.subplots()
            for cluster in df_clientes["Cluster"].unique():
                cluster_data = df_clientes[df_clientes["Cluster"] == cluster]
                ax.scatter(cluster_data["Idade"], cluster_data["Gasto Médio"], label=f'Cluster {cluster}')
            ax.set_xlabel("Idade")
            ax.set_ylabel("Gasto Médio")
            ax.set_title("Segmentação de Clientes")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("As colunas necessárias para a clusterização não foram encontradas.")
beatriz-cardoso-cunha/)")


