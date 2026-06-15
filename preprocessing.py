# preprocessing.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def get_preprocessor():
    # 1. Separacao manual das variaveis para evitar padronizar dados booleanos
    # Estas sao as unicas que precisam ter a escala ajustada (valores altos/quebrados)
    numeric_features = [
        'age', 
        'creatinine_phosphokinase', 
        'ejection_fraction', 
        'platelets', 
        'serum_creatinine', 
        'serum_sodium'
    ]
    
    # Estas sao as binarias (0 ou 1). O professor alertou sobre elas.
    # Elas nao devem passar pelo StandardScaler.
    binary_features = [
        'anaemia', 
        'diabetes', 
        'high_blood_pressure', 
        'sex', 
        'smoking'
    ]

    # 2. Construcao do Pipeline dividindo o tratamento
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', 'passthrough', binary_features) # 'passthrough' = ignora/nao altera
        ])
    
    return preprocessor