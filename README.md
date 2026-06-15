# QUESTÃO 1: Heart Failure

## a) Justificativa do Metaestimador

O projeto adota o RandomForestClassifier devido à sua robustez no tratamento de conjuntos de dados clínicos de pequeno porte (299 amostras). Por ser um modelo baseado em ensemble (bagging), reduz significativamente a variância e o risco de overfitting. Além disso, o metaestimador permite a extração da importância de variáveis, essencial para a explicabilidade médica (ex: identificação de que ejection_fraction e serum_creatinine são preditores críticos). A implementação inclui a busca por hiperparâmetros via GridSearchCV, garantindo uma configuração otimizada que supera heurísticas de tentativa e erro.

## b) Procedimentos de Pré-processamento

O pipeline foi configurado para tratar a heterogeneidade dos dados:

* **Tratamento de Variáveis Binárias:** Através de um ColumnTransformer, as variáveis categóricas binárias (anaemia, diabetes, etc.) foram submetidas a um passthrough, preservando seu significado lógico (0 ou 1). Em paralelo, variáveis numéricas receberam o RobustScaler, garantindo que diferenças de escala não enviesassem o modelo.
* **Prevenção de Data Leakage:** A variável time (tempo de acompanhamento) foi removida do conjunto de treino. Esta medida é crítica, pois tal variável representa informações temporais pós-evento que não estariam disponíveis no momento da predição em um cenário real.

## c) Inferência em Funcionamento

O sistema utiliza o modelo persistido em `heart_failure_model.pkl`. O módulo de inferência processa os dados do paciente, aplica o pipeline de pré-processamento e retorna a classificação de risco acompanhada de uma probabilidade calculada via predict_proba. A saída é consolidada tanto no terminal quanto por meio de um gráfico visual (`graficos/02_risco_paciente.png`), facilitando a interpretação diagnóstica.
