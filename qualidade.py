import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Controle de Qualidade - Usinagem",
    page_icon="⚙️",
    layout="wide"
)

# Título da Aplicação
st.title("⚙️ Controle de Qualidade de Peças de Usinagem")
st.markdown("Aplicação para classificação automática de peças baseada em peso e espessura utilizando **Support Vector Machine (SVM)**.")

# ---------------------------------------------------------
# 1. GERAÇÃO DE DADOS SINTÉTICOS E TREINAMENTO DO MODELO
# ---------------------------------------------------------

@st.cache_resource
def treinar_modelo():
    # Gerando dados sintéticos com 3 classes
    X_synthetic, y_synthetic = make_classification(
        n_samples=600,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )

    # Ajustando a escala dos dados sintéticos para valores realistas de peso (g) e espessura (mm)
    X_scaled = X_synthetic.copy()
    X_scaled[:, 0] = X_synthetic[:, 0] * 15 + 250   # Peso_gramas
    X_scaled[:, 1] = X_synthetic[:, 1] * 2 + 15      # Espessura_mm

    X = pd.DataFrame(X_scaled, columns=['Peso_gramas', 'Espessura_mm'])
    y = y_synthetic

    # Separação em dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Treinamento do modelo SVM
    model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Nomes das classes
    class_names = ['Aprovada', 'Retrabalho', 'Refugo']

    return model, X, y, X_test, y_test, class_names

model, X, y, X_test, y_test, class_names = treinar_modelo()

# ---------------------------------------------------------
# 2. INTERFACE DO USUÁRIO (SIDEBAR / PAINEL)
# ---------------------------------------------------------

st.sidebar.header("📋 Inspeção de Peça")
st.sidebar.write("Insira as medições da peça inspecionada:")

# Definindo limites baseados nos dados treinados
min_peso, max_peso = float(X['Peso_gramas'].min()), float(X['Peso_gramas'].max())
mean_peso = float(X['Peso_gramas'].mean())

min_esp, max_esp = float(X['Espessura_mm'].min()), float(X['Espessura_mm'].max())
mean_esp = float(X['Espessura_mm'].mean())

# Sliders para entrada do inspetor
peso_input = st.sidebar.slider(
    "Peso da Peça (gramas)", 
    min_value=round(min_peso, 1), 
    max_value=round(max_peso, 1), 
    value=round(mean_peso, 1),
    step=0.1
)

espessura_input = st.sidebar.slider(
    "Espessura da Peça (mm)", 
    min_value=round(min_esp, 1), 
    max_value=round(max_esp, 1), 
    value=round(mean_esp, 1),
    step=0.1
)

btn_avaliar = st.sidebar.button("🔍 Avaliar Peça", use_container_width=True)

# ---------------------------------------------------------
# 3. LAYOUT PRINCIPAL (RESULTADOS E MATRIZ DE CONFUSÃO)
# ---------------------------------------------------------

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Resultado da Avaliação")
    
    if btn_avaliar:
        # Predição da peça inspecionada
        nova_peca = pd.DataFrame([[peso_input, espessura_input]], columns=['Peso_gramas', 'Espessura_mm'])
        predicao_idx = model.predict(nova_peca)[0]
        resultado_classe = class_names[predicao_idx]

        # Exibição do status com base no resultado
        if resultado_classe == 'Aprovada':
            st.success(f"### Status: ✅ APROVADA\nA peça atende a todos os requisitos de qualidade.")
        elif resultado_classe == 'Retrabalho':
            st.warning(f"### Status: ⚠️ RETRABALHO\nA peça está fora de tolerância, mas pode ser reusinada.")
        else: # Refugo
            st.error(f"### Status: ❌ REFUGO\nA peça está fora do padrão e deve ser descartada.")

        st.write(f"**Valores Avaliados:**")
        st.write(f"- **Peso:** {peso_input:.2f} g")
        st.write(f"- **Espessura:** {espessura_input:.2f} mm")
    else:
        st.info("Ajuste os parâmetros na barra lateral e clique no botão **'Avaliar Peça'** para classificar.")

with col2:
    st.subheader("📈 Matriz de Confusão do Modelo (SVM)")
    
    # Avaliação do modelo no conjunto de teste
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Plotando a Matriz de Confusão
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=False,
        ax=ax
    )
    ax.set_xlabel('Classe Preditiva')
    ax.set_ylabel('Classe Real')
    ax.set_title('Matriz de Confusão (Dados de Teste)')
    
    st.pyplot(fig)

# ---------------------------------------------------------
# 4. EXIBIÇÃO DO DATAFRAME ABAIXO DOS GRÁFICOS
# ---------------------------------------------------------

st.markdown("---")
st.subheader("📋 Amostras e Resultados do Conjunto de Teste")
st.markdown("Abaixo estão listadas as peças utilizadas para validação do modelo, comparando os valores reais com as predições do algoritmo SVM.")

# Criando um DataFrame consolidado do conjunto de teste com as predições
df_resultados_teste = X_test.copy()
df_resultados_teste['Peso_gramas'] = df_resultados_teste['Peso_gramas'].round(2)
df_resultados_teste['Espessura_mm'] = df_resultados_teste['Espessura_mm'].round(2)
df_resultados_teste['Classe Real'] = [class_names[c] for c in y_test]
df_resultados_teste['Classe Preditiva'] = [class_names[c] for c in y_pred]

# Reordenando e resetando o índice para melhor visualização
df_resultados_teste = df_resultados_teste.reset_index(drop=True)

# Exibindo a tabela interativa no Streamlit
st.dataframe(df_resultados_teste, use_container_width=True, height=300)
