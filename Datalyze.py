import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind, f_oneway

# ======================================
# CONFIGURAÇÃO INICIAL
# ======================================
st.set_page_config(page_title="Datalyze - Análise Inteligente", layout="wide")

st.title("📊 Datalyze - Análise Inteligente de Negócios")
st.write("""
**Bem-vindo!** Carregue seus dados e descubra insights poderosos para seu negócio através de:
- 🔮 Previsões de vendas
- 👥 Segmentação de clientes
- 📈 Comparações estatísticas
""")

# ======================================
# FUNÇÕES PRINCIPAIS
# ======================================

def carregar_dados():
    """Carrega e processa os dados do usuário"""
    uploaded_file = st.sidebar.file_uploader("📤 Carregar arquivo (CSV/XLSX)", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            if 'data' in df.columns:
                df['data'] = pd.to_datetime(df['data'])
                min_date = df['data'].min().date()
                max_date = df['data'].max().date()
                
                st.sidebar.subheader("🗓️ Filtro Temporal")
                start, end = st.sidebar.date_input("Selecione o período:", [min_date, max_date])
                df = df[(df['data'] >= pd.Timestamp(start)) & (df['data'] <= pd.Timestamp(end))]
            
            st.session_state.df = df
            return df
        
        except Exception as e:
            st.error(f"❌ Erro ao ler arquivo: {str(e)}")
            return None
    return None

def analise_clusters(df):
    """Realiza e explica a clusterização para leigos"""
    st.write("""
    ### 👥 Análise de Segmentação de Clientes
    **Como funciona:** 
    Agrupamos automaticamente seus clientes em 3 perfis com base em:
    - Idade
    - Frequência de compras
    - Valor médio gasto
    """)
    
    try:
        # Modelagem
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[['idade', 'frequencia_compra', 'gasto_medio']])
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(10,6))
        cores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for cluster in range(3):
            dados = df[df['cluster'] == cluster]
            ax.scatter(dados['idade'], dados['gasto_medio'], 
                      s=100, c=cores[cluster], 
                      label=f'Grupo {cluster+1}', alpha=0.7)
        
        ax.set_title('Perfil dos Clientes', pad=20)
        ax.set_xlabel('Idade', labelpad=10)
        ax.set_ylabel('Gasto Médio (R$)', labelpad=10)
        ax.legend(title='Segmentos:')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
        
        # Explicação dos clusters
        st.subheader("📌 Características de Cada Grupo")
        
        resumo = df.groupby('cluster').agg({
            'idade': 'mean',
            'frequencia_compra': 'mean',
            'gasto_medio': ['mean', 'std']
        }).reset_index()
        
        resumo.columns = ['Grupo', 'Idade Média', 'Frequência Média', 'Gasto Médio', 'Variação no Gasto']
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(resumo.style.format({
                'Idade Média': '{:.1f} anos',
                'Frequência Média': '{:.1f} compras/mês',
                'Gasto Médio': 'R${:.2f}',
                'Variação no Gasto': '± {:.2f}'
            }))
        
        with col2:
            st.write("""
            **Como interpretar:**
            1. **Grupo 1 (Vermelho):**  
               - Clientes mais jovens  
               - Menor fidelidade  
               - Ideal para campanhas de atração
            
            2. **Grupo 2 (Verde):**  
               - Clientes de média idade  
               - Maior frequência de compras  
               - Foco em fidelização
            
            3. **Grupo 3 (Azul):**  
               - Clientes mais maduros  
               - Maior valor médio gasto  
               - Priorizar experiência premium
            """)
        
        return df
    
    except Exception as e:
        st.error(f"Erro na análise: {str(e)}")
        return None

def previsao_vendas(df, granularidade):
    """Gera previsões com explicação simplificada"""
    st.write(f"""
    ### 🔮 Previsão de Vendas ({granularidade})
    **Metodologia:**
    - Analisamos padrões históricos
    - Calculamos tendência usando inteligência artificial
    - Projeção para os próximos períodos
    """)
    
    try:
        # Processamento
        freq = 'M' if granularidade == "Mês" else 'W' if granularidade == "Semana" else 'D'
        df['periodo'] = df['data'].dt.to_period(freq).dt.to_timestamp()
        
        df_agrupado = df.groupby('periodo', as_index=False).agg({'vendas': 'sum'})
        df_agrupado['dias'] = (df_agrupado['periodo'] - df_agrupado['periodo'].min()).dt.days
        
        # Modelagem
        model = LinearRegression()
        model.fit(df_agrupado[['dias']], df_agrupado['vendas'])
        df_agrupado['previsao'] = model.predict(df_agrupado[['dias']])
        
        # Plot
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df_agrupado['periodo'], df_agrupado['vendas'], 'o-', label='Vendas Reais')
        ax.plot(df_agrupado['periodo'], df_agrupado['previsao'], '--', color='red', label='Tendência')
        
        date_format = '%b/%Y' if granularidade == "Mês" else '%d/%m' if granularidade == "Semana" else '%d/%m/%Y'
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.xticks(rotation=45)
        ax.set_title(f"Evolução das Vendas - {granularidade}")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.write("""
        **Interpretação:**
        - Linha sólida: Valores históricos reais
        - Linha tracejada: Tendência identificada
        - Use para planejar estoque e equipe
        """)
    
    except Exception as e:
        st.error(f"Erro na previsão: {str(e)}")

# ======================================
# INTERFACE PRINCIPAL
# ======================================

# Menu lateral
st.sidebar.title("⚙️ Configurações")
analise = st.sidebar.selectbox(
    "Escolha sua análise:",
    ["Previsão de Vendas", "Segmentação de Clientes", "Comparação de Grupos"]
)

df = carregar_dados()

# Controles de análise
if df is not None:
    st.subheader("📋 Base de Dados Carregada")
    st.dataframe(df.head().style.format({"data": lambda t: t.strftime("%d/%m/%Y")}))
    
    if 'data' in df.columns:
        granularidade = st.sidebar.selectbox(
            "Período de análise:", 
            ["Dia", "Semana", "Mês"], 
            index=1
        )

    # Execução das análises
    if analise == "Previsão de Vendas":
        if {'data', 'vendas'}.issubset(df.columns):
            previsao_vendas(df, granularidade)
        else:
            st.warning("⚠️ Necessário colunas 'data' e 'vendas'")
    
    elif analise == "Segmentação de Clientes":
        if {'idade', 'frequencia_compra', 'gasto_medio'}.issubset(df.columns):
            analise_clusters(df)
        else:
            st.warning("⚠️ Necessário colunas: idade, frequencia_compra, gasto_medio")
    
    elif analise == "Comparação de Grupos":
        # (Implementação similar para testes estatísticos)
        pass

# Rodapé
st.markdown("---")
st.markdown("""
**📬 Suporte:**  
Beatriz Cardoso Cunha  
📧 [beacarcun@gmail.com](mailto:beacarcun@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/beatriz-cardoso-cunha/)
""")
