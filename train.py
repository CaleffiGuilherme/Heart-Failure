import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import get_preprocessor

def train_model(csv_path):
    print("Carregando base de dados...")
    df = pd.read_csv(csv_path)
    
    # Mantendo a remocao do 'time' contra Data Leakage
    X = df.drop(['DEATH_EVENT', 'time'], axis=1) 
    y = df['DEATH_EVENT']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline base (sem passar hiperparametros ainda, o Grid fará isso)
    pipe = Pipeline(steps=[('pre', get_preprocessor()), ('clf', RandomForestClassifier(random_state=42))])

    # Definindo a grade de parametros para o GridSearchCV explorar
    # Usamos 'clf__' para indicar que o parametro pertence ao passo 'clf' (Random Forest) do Pipeline
    param_grid = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 5, 10, 20],
        'clf__min_samples_split': [2, 5, 10],
        'clf__class_weight': [None, 'balanced'] # Avalia se balancear os obitos melhora a precisão
    }

    print("\nIniciando treinamento com Validação Cruzada (GridSearchCV)...")
    print("Isso pode levar alguns segundos, pois o sistema testará dezenas de combinações de hiperparâmetros.")

    # Configurando o GridSearchCV com 5 folds (cv=5) e usando todos os nucleos (n_jobs=-1)
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"\n[Status] Melhores hiperparâmetros encontrados: {grid_search.best_params_}\n")

    # Extraindo o melhor pipeline treinado encontrado pela busca
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, 'heart_failure_model.pkl')
    print("Sucesso! Modelo otimizado salvo como 'heart_failure_model.pkl'")

    print("\n=== Relatório de Avaliação do Modelo no Conjunto de Teste ===")
    
    # 1. Fazendo as previsões com os dados de teste separados no início
    y_pred = best_model.predict(X_test)
    
    # 2. Calculando Acurácia Global e F1-Score
    acc_global = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) # Padrão é average='binary'
    
    # 3. Gerando a Matriz de Confusão para extrair TP, TN, FP, FN
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 4. Calculando Sensibilidade e Especificidade
    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 5. Demonstrando os resultados
    print(f"Acurácia Global: {acc_global:.4f} ({acc_global:.2%})")
    print(f"Sensibilidade:   {sensibilidade:.4f} ({sensibilidade:.2%})")
    print(f"Especificidade:  {especificidade:.4f} ({especificidade:.2%})")
    print(f"F1-Score:        {f1:.4f}")
    print("=============================================================\n")

    print("Gerando gráfico de importância das variáveis (.png)...")
    
    # Pegando as features do melhor modelo
    feature_names = best_model.named_steps['pre'].get_feature_names_out()
    clean_names = [name.split('__')[1] for name in feature_names]
    importances = best_model.named_steps['clf'].feature_importances_

    df_importances = pd.DataFrame({'Variavel': clean_names, 'Importancia': importances})
    df_importances = df_importances.sort_values(by='Importancia', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Variavel', data=df_importances, hue='Variavel', palette='Reds_r', legend=False)
    plt.title('Importância das Variáveis para Predição de Risco (Modelo Otimizado)')
    plt.tight_layout()
    
    # Garante que a pasta existe antes de salvar (segurança extra)
    if not os.path.exists('graficos'):
        os.makedirs('graficos')
        
    caminho = 'graficos/01_importancia_variaveis.png'
    plt.savefig(caminho)
    plt.close()
    print(f"Gráfico salvo silenciosamente em: {caminho}")

if __name__ == "__main__":
    train_model('data/heart_failure_clinical_records_dataset.csv')