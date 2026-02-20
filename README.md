# 🛡️ Detecção de Fraudes Financeiras: Uma Abordagem de Pesquisa em IA Explicável

## 📖 Introdução e Mentalidade de Pesquisa
Neste projeto, assumo o papel de Pesquisador em IA para o desafio individual da **Liga Acadêmica de Inteligência Artificial (Ligia - UFPE)**. Meu objetivo central não foi apenas alcançar um score elevado no Leaderboard, mas construir uma solução robusta, reprodutível e totalmente explicável, transformando o que poderia ser uma "caixa-preta" em um sistema transparente e fundamentado.

A detecção de fraudes é um problema clássico de **dados altamente desbalanceados** (onde as fraudes são eventos raríssimos). Para enfrentar esse desafio, apliquei técnicas de engenharia de features matemáticas, otimização bayesiana e interpretação via Teoria dos Jogos.

---

## 🏗️ 1. Estrutura do Projeto
Para garantir a qualidade de engenharia e a reprodutibilidade exigidas, organizei o repositório de forma modular:

* **`data/`**: Contém os conjuntos de dados `train.csv` e `test.csv` (protegidos via `.gitignore`).
* **`src/preprocessing.py`**: Módulo contendo a lógica de limpeza, transformação cíclica e escalonamento.
* **`src/model.py`**: Implementação da otimização de hiperparâmetros (Optuna) e treinamento do XGBoost.
* **`notebooks/main.ipynb`**: Notebook de execução, análise exploratória e geração de gráficos de explicabilidade.
* **`requirements.txt`**: Lista de dependências para garantir que o ambiente seja idêntico em qualquer máquina.

---

## 🔬 2. Metodologia e Decisões Técnicas

### 2.1 Engenharia de Features: O Tratamento Cíclico do Tempo
Uma das minhas principais decisões foi o tratamento da variável `Time`. Em vez de tratá-la como um contador linear de segundos, eu a transformei em coordenadas de **Seno e Cosseno** ($sin(t)$ e $cos(t)$).
* **Por que?** Em um contador linear, as 23:59h e as 00:01h parecem distantes numericamente, quando na verdade são vizinhas. Ao mapear o tempo em um círculo unitário, eu permito que o modelo capture padrões de sazonalidade (como fraudes que ocorrem mais frequentemente de madrugada) de forma contínua e natural.

### 2.2 Escalonamento Robusto e Prevenção de *Data Leakage*
Para a variável `Amount` (valor da transação), optei pelo **`RobustScaler`**.
* **Por que?** Fraudes costumam apresentar valores discrepantes (outliers). O `RobustScaler` utiliza a mediana e o intervalo interquartil, tornando o escalonamento imune a esses outliers que poderiam distorcer uma normalização padrão.
* **Rigor Científico:** Implementei uma lógica rigorosa para evitar o **Vazamento de Dados (Data Leakage)**. Eu treinei o escalonador apenas nos dados de treino (`fit_transform`) e utilizei esse "molde" apenas para transformar os dados de validação e teste (`transform`), garantindo que nenhuma informação do futuro influenciasse o aprendizado.

---

## 🤖 3. Modelagem e Otimização

### 3.1 XGBoost vs. Outras Arquiteturas
Embora o material de apoio discuta Random Forests (que usam *Bagging*), eu escolhi o **XGBoost (Gradient Boosting)**.
* **Fundamentação:** O Gradient Boosting é sequencial: cada nova árvore de decisão foca especificamente em corrigir os erros residuais das árvores anteriores. Em um problema onde a fraude é a "agulha no palheiro", essa natureza de correção de erros sequencial é superior à votação independente das Random Forests.

### 3.2 Otimização Bayesiana com Optuna
Em vez de testar parâmetros manualmente, utilizei o **Optuna** para realizar uma busca inteligente no espaço de hiperparâmetros.
* **scale_pos_weight:** O parâmetro mais crítico. O Optuna encontrou um valor de aproximadamente **89.8**, o que significa que o modelo dá um peso quase 90 vezes maior para a classe de fraudes, compensando matematicamente o desbalanceamento sem a necessidade de criar dados sintéticos (SMOTE).

---

## 📊 4. Resultados e Métricas de Negócio

### 4.1 ROC-AUC: O Critério de Avaliação
Conforme o edital, otimizei o modelo para a métrica **ROC-AUC**. Meu modelo atingiu um score de **0.9872** na validação local, demonstrando uma altíssima capacidade de ordenar transações por risco.

### 4.2 Métricas de Negócio (Recall e Precisão)
No meu relatório técnico, decidi não olhar apenas para a probabilidade, mas sim para o impacto real. Ajustando o limiar de decisão (*threshold*) para **0.3**, obtive:
* **Recall de 80%:** Identificamos 8 em cada 10 fraudes.
* **Precisão de 88%:** Mantivemos o erro de bloquear clientes legítimos em um nível muito baixo.

---

## 🔎 5. Explicabilidade (XAI) com SHAP
Para garantir que o modelo não seja uma "Caixa-Preta" (exigência do edital), utilizei o **SHAP (SHapley Additive exPlanations)**.
* **Análise Global:** O gráfico `summary_plot` revelou que as variáveis **V4, V14 e V12** são as mais influentes. Valores baixos em V14 e V12 aumentam drasticamente a suspeita de fraude.
* **Análise Local:** Gere gráficos de cascata (*Waterfall*) para explicar transações individuais, provando exatamente quais características levaram o modelo a considerar aquela operação específica como fraudulenta.

---

## 🏁 Conclusão e Reprodutibilidade
Para garantir a integridade científica, fixei a semente aleatória (**seed/random_state**) em **42** em todas as etapas, desde a separação dos dados até o treinamento do XGBoost, conforme solicitado pelo edital.

Este trabalho representa uma busca contínua em unir a engenharia de software à pesquisa científica em IA com precisão, motivando-me a entregar uma solução que não apenas performa, mas que é justificável e segura.