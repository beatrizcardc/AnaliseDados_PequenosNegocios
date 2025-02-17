import streamlit as st
from datetime import datetime, timedelta
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

# Lista de feriados nacionais de 2024 a 2026
feriados_nacionais = [
    "2024-01-01", "2024-04-21", "2024-05-01", "2024-09-07", "2024-10-12", "2024-11-02", "2024-11-15", "2024-12-25",
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-09-07", "2025-10-12", "2025-11-02", "2025-11-15", "2025-12-25",
    "2026-01-01", "2026-04-21", "2026-05-01", "2026-09-07", "2026-10-12", "2026-11-02", "2026-11-15", "2026-12-25"
]
feriados_nacionais = [pd.Timestamp(date) for date in feriados_nacionais]

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

# Função de previsão de vendas
def prever_vendas(df):
    if {'dia_semana', 'horario', 'temperatura', 'vendas', 'categoria_produto'}.issubset(df.columns):
        df['categoria_produto'] = df['categoria_produto'].astype(str)
        df['categoria_produto'] = df['categoria_produto'].astype('category').cat.codes + 1  # Classifica de 1 a 5
        produto_selecionado = st.sidebar.selectbox("Escolha um produto para prever vendas (1-5):", sorted(df['categoria_produto'].unique()))
        df_filtrado = df[df['categoria_produto'] == produto_selecionado]
        
        X = df_filtrado[['dia_semana', 'horario', 'temperatura']]
        y = df_filtrado['vendas']
        modelo = LinearRegression().fit(X, y)
        df_filtrado['previsao_vendas'] = modelo.predict(X)
        return df_filtrado, modelo, produto_selecionado
    else:
        st.warning("O arquivo precisa conter as colunas: dia_semana, horario, temperatura, vendas, categoria_produto. Por favor, verifique se selecionou a planilha correta. Para a análise de previsão de vendas, selecione a planilha de 'Vendas'.")
        return None, None, None

# Função de clusterização
def clusterizar_clientes(df):
    if {'idade', 'frequencia_compra', 'gasto_medio'}.issubset(df.columns):
        kmeans = KMeans(n_clusters=3, random_state=42).fit(df[['idade', 'frequencia_compra', 'gasto_medio']])
        df['cluster'] = kmeans.labels_
        return df
    else:
        st.warning("O arquivo precisa conter as colunas: idade, frequencia_compra, gasto_medio. Por favor, verifique se selecionou a planilha correta. Para a análise de clusterização de clientes, selecione a planilha de 'Clientes'.")
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

