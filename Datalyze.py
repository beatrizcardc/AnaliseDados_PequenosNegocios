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

# Função de clusterização com visualização
def clusterizar_clientes(df):
    # Verifica colunas necessárias
    if not {'idade', 'frequencia_compra', 'gasto_medio'}.issubset(df.columns):
        st.warning("""O arquivo precisa conter as colunas: 
                   idade, frequencia_compra, gasto_medio. 
                   Verifique se selecionou a planilha de 'Clientes'.""")
        return None
    
    try:
        # Clusterização
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[['idade', 'frequencia_compra', 'gasto_medio']])
        
        # Criação do gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        cores = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Palette de cores acessível
        
        for cluster in sorted(df['cluster'].unique()):
            dados_cluster = df[df['cluster'] == cluster]
            ax.scatter(
                dados_cluster['idade'], 
                dados_cluster['gasto_medio'],
                s=100,  # Tamanho dos pontos
                c=cores[cluster],
                label=f'Cluster {cluster + 1}',
                alpha=0.7
            )
            
        # Personalização do gráfico
        ax.set_title('Segmentação de Clientes', pad=20)
        ax.set_xlabel('Idade', labelpad=10)
        ax.set_ylabel('Gasto Médio (R$)', labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(title='Grupos')
        
        # Exibição no Streamlit
        st.pyplot(fig)
        
        return df
        
    except Exception as e:
        st.error(f"Erro na clusterização: {str(e)}")
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

    # Seletor de granularidade APENAS se existir coluna 'data'
    if 'data' in df.columns:
        st.sidebar.subheader("🗓️ Configuração do Gráfico")
        granularidade = st.sidebar.selectbox(
            "Agrupar vendas por:",
            ["Dia", "Semana", "Mês"],
            index=1
        )

if analise_selecionada == "Previsão de Vendas":
    if {'data', 'vendas'}.issubset(df.columns):
        try:
            # Converter para datetime e ordenar
            df['data'] = pd.to_datetime(df['data'])
            df = df.sort_values('data')
            
            # Criar coluna de agrupamento temporal
            if granularidade == "Mês":
                df['periodo'] = df['data'].dt.to_period('M').dt.to_timestamp()
            elif granularidade == "Semana":
                df['periodo'] = df['data'].dt.to_period('W').dt.to_timestamp()
            else:  # Dia
                df['periodo'] = df['data']

            # Agregar dados por período
            df_agrupado = df.groupby('periodo', as_index=False).agg({
                'vendas': 'sum',
                'data': 'first'
            })
            
            # Calcular dias desde a primeira data
            df_agrupado['dias'] = (df_agrupado['periodo'] - df_agrupado['periodo'].min()).dt.days
            
            # Modelo de Regressão Linear
            model = LinearRegression().fit(df_agrupado[['dias']], df_agrupado['vendas'])
            df_agrupado['previsao'] = model.predict(df_agrupado[['dias']])
            
            # Plot
            st.write("### 📈 Previsão de Vendas")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plotar dados reais e previsão
            ax.plot(df_agrupado['periodo'], df_agrupado['vendas'], 
                    marker='o', label='Vendas Reais')
            ax.plot(df_agrupado['periodo'], df_agrupado['previsao'], 
                    linestyle='--', color='red', label='Previsão')
            
            # Configurações do gráfico
            ax.set_title(f"Vendas por {granularidade.lower()} - Modelo de Regressão")
            ax.set_xlabel("Período")
            ax.set_ylabel("Vendas (R$)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Formatando datas conforme granularidade
            date_format = '%b/%Y' if granularidade == "Mês" else '%d/%m/%Y' if granularidade == "Dia" else 'Sem. %W/%Y'
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erro na geração da previsão: {str(e)}")
    else:
        st.warning("⚠️ Dados incompletos! Necessário colunas 'data' e 'vendas'.")

    # Análise de Clusterização
    elif analise_selecionada == "Clusterização de Clientes":
        df_clusterizado = clusterizar_clientes(df)
        if df_clusterizado is not None:
            st.write("### Detalhes dos Clusters")
            st.dataframe(
                df_clusterizado.groupby('cluster').agg({
                    'idade': 'mean',
                    'frequencia_compra': 'mean',
                    'gasto_medio': 'mean'
                }).style.format("{:.1f}")
            )

    # Análise Estatística
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

