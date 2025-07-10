import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime

st.set_page_config(layout="centered", page_title="Lotofácil IA v9.2")

st.title("🎯 Loto IA – Além da Sorte, Tem análises por IA ")
st.subheader("IA + Clusterização + Score Híbrido com Filtros Inteligentes")

# Funções
def carregar_base(uploaded_file):
    df = pd.read_excel(uploaded_file, skiprows=6)
    col_dezenas = [f"bola {i}" for i in range(1, 16)]
    df = df[col_dezenas].dropna()
    return df

def preparar_dados_lstm(df):
    binarios = []
    for _, row in df.iterrows():
        binario = [1 if i+1 in row.values else 0 for i in range(25)]
        binarios.append(binario)
    X = np.array(binarios[:-1])
    y = np.array(binarios[1:])
    return X.reshape((X.shape[0], 1, X.shape[1])), y

def treinar_lstm(X, y):
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(25, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X, y, epochs=40, verbose=0)
    return model

def prever_jogo_lstm(model, ultimo_jogo):
    entrada = np.array([1 if i+1 in ultimo_jogo else 0 for i in range(25)])
    entrada = entrada.reshape((1, 1, 25))
    pred = model.predict(entrada, verbose=0)[0]
    indices = np.argsort(pred)[-15:]
    return sorted([i+1 for i in indices])

def gerar_clusters(df, k=5):
    binarios = []
    for _, row in df.iterrows():
        binario = [1 if i+1 in row.values else 0 for i in range(25)]
        binarios.append(binario)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(binarios)
    return kmeans

def jogo_do_cluster(cluster_center):
    indices = np.argsort(cluster_center)[-15:]
    return sorted([i+1 for i in indices])

def score_hibrido(jogo):
    pares = sum(1 for d in jogo if d % 2 == 0)
    moldura = sum(1 for d in jogo if d in [1,2,3,4,5,6,10,15,20,21,22,23,24,25])
    primos = sum(1 for d in jogo if d in [2,3,5,7,11,13,17,19,23])
    soma = sum(jogo)
    criterios = [
        5 <= pares <= 10,
        8 <= moldura <= 12,
        3 <= primos <= 7,
        180 <= soma <= 250
    ]
    return sum(criterios), {"pares": pares, "moldura": moldura, "primos": primos, "soma": soma}

def explicar(jogo):
    score, info = score_hibrido(jogo)
    return f"✔️ Pares: {info['pares']} | Moldura: {info['moldura']} | Primos: {info['primos']} | Soma: {info['soma']} | Score: {score}/4"

# Interface
uploaded_file = st.file_uploader("📥 Importar arquivo Excel com sorteios", type=["xlsx"])
if uploaded_file:
    df = carregar_base(uploaded_file)
    st.success("Base carregada com sucesso!")

    if st.button("🚀 Gerar Jogos Inteligentes"):
        with st.spinner("Treinando IA e analisando padrões..."):
            # Jogo 1 - IA LSTM
            X, y = preparar_dados_lstm(df)
            model = treinar_lstm(X, y)
            ultimo_jogo = df.iloc[-1].values
            jogo_lstm = prever_jogo_lstm(model, ultimo_jogo)

            st.subheader("🎲 Jogo 1 (IA - LSTM com filtro)")
            st.write("Dezenas:", " - ".join(f"{d:02d}" for d in jogo_lstm))
            st.info(explicar(jogo_lstm))

            # Jogo 2 - Cluster (com proteção)
            try:
                df_14_15 = df[df.apply(lambda row: 14 <= sum([1 for n in row if n in df.iloc[-1].values]) <= 15, axis=1)]
                df_cluster_base = df_14_15 if len(df_14_15) >= 5 else df
                num_clusters = min(5, len(df_cluster_base))
                kmeans = gerar_clusters(df_cluster_base, k=num_clusters)
                cluster_jogo = jogo_do_cluster(kmeans.cluster_centers_[0])

                st.subheader("🎲 Jogo 2 (Baseado em Cluster de Vencedores)")
                st.write("Dezenas:", " - ".join(f"{d:02d}" for d in cluster_jogo))
                st.info(explicar(cluster_jogo))

            except Exception as e:
                st.warning("⚠️ Não foi possível gerar o jogo baseado em cluster de vencedores.\n"
                           "Isso pode ter ocorrido por falta de dados compatíveis.")

        st.success("Jogos gerados com sucesso!")

        if st.button("🔁 Realizar nova análise"):
            st.rerun()
