import os
from train import train_model
from inference import predict_new_patient

def main():
    csv_path = 'data/heart_failure_clinical_records_dataset.csv'
    model_path = 'heart_failure_model.pkl'

    print("=== Sistema de Classificação de Risco Cardíaco ===\n")

    # Cria a pasta de gráficos se ela não existir
    if not os.path.exists('graficos'):
        os.makedirs('graficos')

    if not os.path.exists('data'):
        print("[Erro] A pasta 'data' não foi encontrada.")
        return
        
    if not os.path.exists(csv_path):
        print(f"[Erro] O arquivo CSV não foi encontrado em: {csv_path}")
        return

    # Treinamento
    if not os.path.exists(model_path):
        print("[Status] Modelo não encontrado. Iniciando treinamento...")
        train_model(csv_path)
    else:
        print(f"[Status] Utilizando modelo existente: {model_path}")

    # Paciente de teste (SEM a variável 'time' agora)
    print("\n[Status] Executando inferência para paciente de teste...")
    novo_paciente = {
        'age': 65.0,
        'anaemia': 1,
        'creatinine_phosphokinase': 160,
        'diabetes': 1,
        'ejection_fraction': 20,
        'high_blood_pressure': 0,
        'platelets': 327000.0,
        'serum_creatinine': 2.7,
        'serum_sodium': 116,
        'sex': 0,
        'smoking': 0
    }

    predict_new_patient(novo_paciente)

if __name__ == "__main__":
    main()