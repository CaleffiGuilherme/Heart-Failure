import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

def predict_new_patient(patient_data):
    model_path = 'heart_failure_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"[Erro] Modelo '{model_path}' não encontrado. Treine o modelo primeiro.")
        return

    pipeline = joblib.load(model_path)
    df_patient = pd.DataFrame([patient_data])
    
    prediction = pipeline.predict(df_patient)[0]
    probability = pipeline.predict_proba(df_patient)[0][1]

    print("\n--- Resultado da Predição ---")
    if prediction == 1:
        print("Resultado: ALTO RISCO (Evento de óbito provável)")
        cor_grafico = 'darkred'
    else:
        print("Resultado: BAIXO RISCO (Evento de óbito improvável)")
        cor_grafico = 'forestgreen'
    
    prob_percentual = probability * 100
    print(f"Probabilidade calculada: {prob_percentual:.2f}%")
    print("----------------------------\n")

    # --- GERAÇÃO DO GRÁFICO ESTÁTICO (PNG) ---
    print("Gerando painel do paciente (.png)...")
    
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(['Risco de Óbito'], [prob_percentual], color=cor_grafico, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probabilidade (%)')
    ax.set_title('Avaliação do Paciente')
    
    # Adiciona o texto de porcentagem no final da barra
    ax.text(prob_percentual + 1, 0, f'{prob_percentual:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    caminho = 'graficos/02_risco_paciente.png'
    plt.savefig(caminho)
    plt.close() # Sem popups!
    print(f"Gráfico salvo silenciosamente em: {caminho}")