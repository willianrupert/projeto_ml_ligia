import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def feature_engineering(df: pd.DataFrame, scaler=None, is_train: bool = True):
    """
    Transforma dados brutos em features ricas.
    is_train=True: Treina o Scaler e transforma os dados.
    is_train=False: Apenas aplica o Scaler treinado aos novos dados (Evita Data Leakage).
    """
    df_copy = df.copy()
    
    # 1. Tratamento do Tempo (Matemática Cíclica - Sem risco de vazamento)
    seconds_in_day = 24 * 60 * 60
    if 'Time' in df_copy.columns:
        df_copy['sin_time'] = np.sin(2 * np.pi * df_copy['Time'] / seconds_in_day)
        df_copy['cos_time'] = np.cos(2 * np.pi * df_copy['Time'] / seconds_in_day)
    
    # 2. Tratamento do Valor (Evitando Data Leakage)
    if 'Amount' in df_copy.columns:
        if is_train:
            scaler = RobustScaler()
            df_copy['Amount_Scaled'] = scaler.fit_transform(df_copy[['Amount']])
        else:
            if scaler is None:
                raise ValueError("Um scaler treinado deve ser passado para dados de teste!")
            # Apenas transforma, baseado no que aprendeu no treino
            df_copy['Amount_Scaled'] = scaler.transform(df_copy[['Amount']])
    
    # Remove colunas originais
    cols_to_drop = [c for c in ['Time', 'Amount', 'id'] if c in df_copy.columns]
    df_copy.drop(cols_to_drop, axis=1, inplace=True)
    
    # Retorna o dataframe e o scaler (para usarmos no teste depois)
    if is_train:
        return df_copy, scaler
    return df_copy