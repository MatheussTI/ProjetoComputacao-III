import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Modelo de Regressão Linear com Filtros Personalizados")

# File uploader for CSV
uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("### Visualização de dados originais")
    st.write(df.head())

    # Initialize session state for filters
    if 'filters' not in st.session_state:
        st.session_state.filters = []

    # Interface for adding filters
    st.write("### Adicionar filtros personalizados")
    with st.form(key="filter_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_column = st.selectbox("Selecione a coluna para filtrar", df.columns, key="filter_column")
        with col2:
            filter_operation = st.selectbox(
                "Selecione a operação",
                ["is equal to", "is not equal to", "contains", "does not contain", "greater than", "less than"],
                key="filter_operation"
            )
        with col3:
            filter_value = st.text_input("Insira o valor", key="filter_value")
        submit_filter = st.form_submit_button("Adicionar filtro")

    # Add filter to session state
    if submit_filter and filter_column and filter_value:
        st.session_state.filters.append({
            'column': filter_column,
            'operation': filter_operation,
            'value': filter_value
        })

    # Display current filters
    if st.session_state.filters:
        st.write("### Filtros atuais")
        for i, filt in enumerate(st.session_state.filters):
            st.write(f"Filter {i+1}: {filt['column']} {filt['operation']} {filt['value']}")
            if st.button(f"Remove Filter {i+1}", key=f"remove_{i}"):
                st.session_state.filters.pop(i)
                st.rerun()

    # Apply filters
    df_filtered = df.copy()
    try:
        for filt in st.session_state.filters:
            col = filt['column']
            op = filt['operation']
            val = filt['value']
            
            if op == "is equal to":
                df_filtered = df_filtered[df_filtered[col] == val]
            elif op == "is not equal to":
                df_filtered = df_filtered[df_filtered[col] != val]
            elif op == "contains":
                df_filtered = df_filtered[df_filtered[col].astype(str).str.contains(val, case=False, na=False)]
            elif op == "does not contain":
                df_filtered = df_filtered[~df_filtered[col].astype(str).str.contains(val, case=False, na=False)]
            elif op == "greater than":
                df_filtered = df_filtered[df_filtered[col].astype(float) > float(val)]
            elif op == "less than":
                df_filtered = df_filtered[df_filtered[col].astype(float) < float(val)]
    except Exception as e:
        st.error(f"Erro ao aplicar filtro: {str(e)}. Por favor, verifique os tipos e valores das colunas.")
        st.stop()

    st.write(f"### Visualização de dados filtrados (Linhas: {len(df_filtered)})")
    st.write(df_filtered.head())

    # Select columns for X (features) and Y (target)
    st.write("### Selecione Colunas")
    all_columns = df_filtered.columns.tolist()
    
    # Select target column (Y)
    target_column = st.selectbox("Selecione a coluna de previsão (Y)", all_columns, key="target_column")
    
    # Select feature columns (X)
    feature_columns = st.multiselect("Selecione colunas de features (X)", [col for col in all_columns if col != target_column])
    
    # Select columns for one-hot encoding
    categorical_columns = st.multiselect("Selecione colunas para one-hot encoding", all_columns)
    
    if feature_columns and target_column:
        # Prepare the data
        df_cleaned = df_filtered[feature_columns + [target_column]].copy()
        
        # Apply one-hot encoding to selected categorical columns
        if categorical_columns:
            for col in categorical_columns:
                if col in df_cleaned.columns:
                    one_hot = pd.get_dummies(df_cleaned[col], prefix=col)
                    df_cleaned = df_cleaned.drop(col, axis=1)
                    df_cleaned = df_cleaned.join(one_hot)
        
        # Separate features (X) and target (Y)
        Y = df_cleaned[target_column]
        X = df_cleaned.drop(columns=[target_column])
        
        # Display processed data
        st.write("### Dados processados (Após One-Hot Encoding)")
        st.write(df_cleaned.head())
        
        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        # Train the model
        lin_reg = LinearRegression()
        lin_reg.fit(x_train, y_train)
        
        # Make predictions on training set
        preds = lin_reg.predict(x_train)
        
        # Calculate metrics
        lin_mse = mean_squared_error(y_train, preds)
        lin_rmse = np.sqrt(lin_mse)
        lin_r2 = r2_score(y_train, preds)
        
        # Display model evaluation metrics
        st.write("### Avaliação de Modelo (Conjunto de Treinamento)")
        st.write(f"**RMSE**: {lin_rmse:.4f}")
        st.write(f"**R²**: {lin_r2:.4f}")
        
        # Predictions vs Actuals for first 5 samples
        st.write("### Previsões vs. Valores Reais (Primeiros 5 Exemplos de Treinamento)")
        alguns_dados = x_train.iloc[:5]
        alguns_label = y_train.iloc[:5]
        predicoes = lin_reg.predict(alguns_dados).round(2)
        labels = alguns_label.values
        
        # Create DataFrame for comparison
        df_result = pd.DataFrame({
            'Predições': predicoes,
            'Valores reais': labels
        })
        
        st.write(df_result)
        
        # Plot Predictions vs Actuals
        st.write("### Visualização: Previsões vs. Valores Reais")
        fig, ax = plt.subplots()
        sns.barplot(x=df_result.index, y='Predições', data=df_result, alpha=0.7, label='Predições', ax=ax)
        sns.barplot(x=df_result.index, y='Valores reais', data=df_result, alpha=0.7, label='Valores reais', dodge=False, ax=ax)
        ax.set_xlabel('Índice')
        ax.set_ylabel('Preço')
        ax.set_title('Predições vs Valores Reais')
        ax.legend()
        st.pyplot(fig)
        
        # Optional: Display correlation matrix
        st.write("### Matriz de correlação (colunas numéricas)")
        numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        if target_column in numeric_cols:
            correlacao = df_cleaned[numeric_cols].corr()[target_column].sort_values(ascending=False)
            st.write(correlacao)
else:
    st.write("Faça upload de um arquivo CSV para prosseguir.")