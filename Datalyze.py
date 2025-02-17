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

# Função de previsão de vendas
def prever_vendas(df):
    if {'dia_semana', 'horario', 'temperatura', 'vendas'}.issubset(df.columns):
        X = df[['dia_semana', 'horario', 'temperatura']]
        y = df['vendas']
        modelo = LinearRegression().fit(X, y)
        df['previsao_vendas'] = modelo.predict(X)
        return df, modelo
    else:
        st.warning("O arquivo precisa conter as colunas: dia_semana, horario, temperatura, vendas. Por favor, verifique se selecionou a planilha correta. Para a análise de previsão de vendas, selecione a planilha de 'Vendas'.")
        return None, None

# Sidebar
st.sidebar.title("📂 Opções de Análise")
analise_selecionada = st.sidebar.selectbox("Escolha uma análise", ["Previsão de Vendas", "Clusterização de Clientes", "Testes Estatísticos"])
df = carregar_dados()

if df is not None:
    st.write("### 📋 Dados Carregados")
    st.dataframe(df.head())

    if analise_selecionada == "Previsão de Vendas":
        # Adiciona a opção para o usuário escolher a variável para visualização do gráfico
        variavel_grafico = st.sidebar.selectbox("Escolha a variável para visualizar a previsão:", ["horario", "dia_semana", "temperatura"])
        df, modelo = prever_vendas(df)
        if df is not None:
            st.write(f"### 📈 Previsão de Vendas vs. Vendas Reais em função de {variavel_grafico.capitalize()}")
            
            if variavel_grafico == 'dia_semana':
                dias_semana = {7: 'Domingo', 1: 'Segunda', 2: 'Terça', 3: 'Quarta', 4: 'Quinta', 5: 'Sexta', 6: 'Sábado'}
                df['dia_semana'] = df['dia_semana'].map(dias_semana)
                df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=['Domingo', 'Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado'], ordered=True)
            
            df_plot = df[[variavel_grafico, 'vendas', 'previsao_vendas']].groupby(variavel_grafico).mean()
            st.line_chart(df_plot)

    st.sidebar.button("🗑️ Limpar Dados", on_click=lambda: st.session_state.pop('df', None))


# Rodapé
st.markdown("---")
st.markdown("**📧 Contato:** Beatriz Cardoso Cunha | Email: beacarcun@gmail.com | LinkedIn: https://www.linkedin.com/in/beatriz-cardoso-cunha/")
