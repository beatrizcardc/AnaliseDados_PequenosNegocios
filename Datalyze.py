import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# Lista de feriados nacionais
feriados_nacionais = [
    "2024-01-01", "2024-04-21", "2024-05-01", "2024-09-07", "2024-10-12", "2024-11-02", "2024-11-15", "2024-12-25",
    "2025-01-01", "2025-04-21", "2025-05-01", "2025-09-07", "2025-10-12", "2025-11-02", "2025-11-15", "2025-12-25",
    "2026-01-01", "2026-04-21", "2026-05-01", "2026-09-07", "2026-10-12", "2026-11-02", "2026-11-15", "2026-12-25"
]
feriados_nacionais = [pd.Timestamp(date) for date in feriados_nacionais]

# Função para carregar dados
def carregar_dados():
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV/XLS", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                xls = pd.ExcelFile(uploaded_file)
                planilhas = xls.sheet_names
                sheet_selecionada = st.sidebar.selectbox("Escolha a planilha:", planilhas)
                df = pd.read_excel(xls, sheet_name=sheet_selecionada)
            
            if 'data' in df.columns:
                df['data'] = pd.to_datetime(df['data'])
                data_min, data_max = df['data'].min(), df['data'].max()
                st.sidebar.subheader("📆 Filtro de Período")
                data_inicio, data_fim = st.sidebar.date_input("Selecione o período:", [data_min, data_max])
                df = df[(df['data'] >= pd.Timestamp(data_inicio)) & (df['data'] <= pd.Timestamp(data_fim))]
            
            st.session_state['df'] = df
            return df
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")
            return None
    return None

# Função de clusterização
def clusterizar_clientes(df):
    if not {'idade', 'frequencia_compra', 'gasto_medio'}.issubset(df.columns):
        st.warning("Colunas necessárias não encontradas: idade, frequencia_compra, gasto_medio")
        return None
    
    try:
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[['idade', 'frequencia_compra', 'gasto_medio']])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        cores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for cluster in df['cluster'].unique():
            dados = df[df['cluster'] == cluster]
            ax.scatter(dados['idade'], dados['gasto_medio'], 
                      s=100, c=cores[cluster], 
                      label=f'Cluster {cluster+1}', alpha=0.7)
        
        ax.set_title('Segmentação de Clientes', pad=20)
        ax.set_xlabel('Idade', labelpad=10)
        ax.set_ylabel('Gasto Médio (R$)', labelpad=10)
        ax.legend(title='Grupos')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
        
        return df
    except Exception as e:
        st.error(f"Erro na clusterização: {str(e)}")
        return None

# Função de testes estatísticos
def testes_estatisticos(df):
    if not {'grupo', 'vendas'}.issubset(df.columns):
        return None, None, ""
    
    grupos = df.groupby('grupo')['vendas'].apply(list)
    if len(grupos) < 2:
        return None, None, ""
    
    try:
        if len(grupos) == 2:
            stat, p = ttest_ind(grupos.iloc[0], grupos.iloc[1])
            return "Teste T", p, "Comparação entre médias de dois grupos independentes"
        else:
            stat, p = f_oneway(*grupos)
            return "ANOVA", p, "Comparação entre médias de três ou mais grupos"
    except:
        return None, None, ""

# Interface principal
st.sidebar.title("📂 Opções de Análise")
analise_selecionada = st.sidebar.selectbox(
    "Escolha uma análise", 
    ["Previsão de Vendas", "Clusterização de Clientes", "Testes Estatísticos"]
)

df = carregar_dados()

if df is not None:
    st.write("### 📋 Dados Carregados")
    st.dataframe(df.head().style.format({"data": lambda t: t.strftime("%d/%m/%Y")}))
    
    # Configuração de granularidade
    if 'data' in df.columns:
        st.sidebar.subheader("🗓️ Configuração Temporal")
        granularidade = st.sidebar.selectbox(
            "Agrupar dados por:",
            ["Dia", "Semana", "Mês"],
            index=1
        )

    # Execução das análises
    if analise_selecionada == "Previsão de Vendas":
        if {'data', 'vendas'}.issubset(df.columns):
            try:
                df = df.sort_values('data')
                df['periodo'] = df['data'].dt.to_period(
                    'M' if granularidade == "Mês" else 'W' if granularidade == "Semana" else 'D'
                ).dt.to_timestamp()
                
                df_agrupado = df.groupby('periodo', as_index=False).agg({
                    'vendas': 'sum',
                    'data': 'first'
                })
                
                df_agrupado['dias'] = (df_agrupado['periodo'] - df_agrupado['periodo'].min()).dt.days
                model = LinearRegression().fit(df_agrupado[['dias']], df_agrupado['vendas'])
                df_agrupado['previsao'] = model.predict(df_agrupado[['dias']])
                
                fig, ax = plt.subplots(figsize=(12,6))
                ax.plot(df_agrupado['periodo'], df_agrupado['vendas'], 'o-', label='Vendas Reais')
                ax.plot(df_agrupado['periodo'], df_agrupado['previsao'], '--', color='red', label='Previsão')
                
                date_format = '%b/%Y' if granularidade == "Mês" else '%d/%m' if granularidade == "Semana" else '%d/%m/%Y'
                ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
                plt.xticks(rotation=45)
                ax.set_title(f"Previsão de Vendas - {granularidade}")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Erro na análise: {str(e)}")
        else:
            st.warning("⚠️ Dados incompletos para análise de vendas!")

    elif analise_selecionada == "Clusterização de Clientes":
        df_cluster = clusterizar_clientes(df)
        if df_cluster is not None:
            st.write("### 📌 Características dos Clusters")
            resumo = df_cluster.groupby('cluster').agg({
                'idade': ['mean', 'std'],
                'frequencia_compra': ['mean', 'std'],
                'gasto_medio': ['mean', 'std']
            })
            st.dataframe(resumo.style.format("{:.1f}"))

    elif analise_selecionada == "Testes Estatísticos":
        teste, p, explicacao = testes_estatisticos(df)
        if teste:
            st.write(f"### 📊 Resultado do {teste}")
            st.metric("p-valor", f"{p:.4f}")
            st.write(f"**Interpretação:** {explicacao}")
            if p < 0.05:
                st.success("Diferença estatisticamente significativa (p < 0.05)")
            else:
                st.info("Nenhuma diferença significativa detectada (p ≥ 0.05)")

    # Botão para limpar dados
    st.sidebar.button("Limpar Dados", on_click=lambda: st.session_state.pop('df'))

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido por:** Beatriz Cardoso Cunha  \n"
            "📧 [beacarcun@gmail.com](mailto:beacarcun@gmail.com)  \n"
            "🔗 [LinkedIn](https://www.linkedin.com/in/beatriz-cardoso-cunha/)")
