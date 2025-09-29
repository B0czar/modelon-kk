# modelon-kk

## sobre o que fazer

Entregáveis
Notebook Completo: Um notebook Jupyter documentando todo o processo, desde a exploração dos dados até a criação e avaliação do modelo. Você pode trabalhar com o notebook dentro da plataforma Kaggle ou importar um arquivo .ipynb
Arquivo CSV de resultados: Submeta os resultados em csv do seu melhor modelo treinado, conforme template disponibilizado.
Regras do Campeonato
Utilize seu e-mail do Inteli na competição, para que a gente possa identificar você e sua entrega.
Sua participação deve ser individual!
Utilize Python e apenas as bibliotecas padrão do módulo: Numpy, Pandas, ScikitLearn.
Encorajamos o uso de bibliotecas de visualização e gráficos para fortalecer suas análises e justificar suas escolhas. Para isso, utilize bibliotecas como: Matplotlib, Seaborn e/ou Plotly.
Não é permitido utilizar outras bibliotecas! Caso queira implementar algoritmos mais avançados, deverá fazê-lo apenas com as ferramentas permitidas.
Não é permitido usar dados externos além do fornecido.
O ranqueamento será dado conforme a performance do seu modelo na métrica de acurácia. Quanto maior, melhor! Ao final, a pessoa que ficar em primeiro lugar na turma ganhará um prêmio (surpresa!). Haverá também prêmio exclusivo para a melhor acurácia dentre todas as turmas de primeiro ano! (critérios de desempate: outras métricas como precisão e recall, além da nota final da entrega)
Não trapaceie! Se seu código possuir semelhança a alguma outra solução pronta, você será desclassificado(a) e ficará com nota zero! (sujeito a sanções disciplinares previstas no regulamento do Inteli)
Critérios de Avaliação das Submissões
Seu trabalho será avaliado com base nos seguintes critérios:

Limpeza e Tratamento de Valores Nulos (até 0,5 pt):
A qualidade dos dados é crucial. Demonstre seu processo de limpeza, incluindo a maneira como lida com valores ausentes e outliers que possam distorcer os resultados.
Codificação de Variáveis Categóricas (até 0,5 pt):
Aplique técnicas apropriadas de codificação para transformar variáveis categóricas em formatos utilizáveis em modelos preditivos, garantindo que a informação essencial não seja perdida no processo.
Exploração e Visualização dos Dados (até 2,0 pts):
Realize uma análise exploratória detalhada para descobrir padrões, correlações e tendências nos dados. Use visualizações eficazes para comunicar seus insights e justificar suas escolhas de features e modelos.
Formulação de Hipóteses (até 1,0 pt):
Formule três hipóteses que possam explicar os fatores que influenciam o sucesso da empresas. Por exemplo, pode-se investigar se a empresas com mais funcionários ou com menos tempo de fundação têm maior chance de sucesso.
Seleção de Features (até 1,0 pt):
Escolha as features mais relevantes para o modelo com base em sua análise exploratória e hipóteses formuladas.
Construção e Avaliação do Modelo (até 2,0 pts):
Selecione um modelo de machine learning adequado (ou uma combinação de modelos) que maximize a capacidade preditiva. A avaliação deve incluir métricas como acurácia, precisão, recall, e F1-score.
Finetuning de Hiperparâmetros (até 1,0 pt):
Realize um ajuste fino (finetuning) dos hiperparâmetros do modelo para otimizar seu desempenho. Detalhe o processo de busca e as justificativas para as escolhas feitas.
Acurácia Mínima (até 2,0 pts):
O modelo deve atingir uma acurácia mínima de 80% para ser considerado bem-sucedido (pontuação total). Embora a acurácia seja a métrica principal usada na competição, analise também outras métricas como precisão e recall, para melhor interpretação do desempenho do modelo preditivo treinado.
Documentação e Apresentação dos Resultados (demérito de até 2,0 pts):
A documentação clara e a apresentação dos resultados são importantes. O notebook final deve ser bem organizado, com código limpo, e o raciocínio por trás de cada decisão deve ser explicado de forma objetiva e compreensível em células de texto, sem exageros.


## sobre os dados

Contexto
Este conjunto de dados reúne informações reais sobre startups de diferentes setores, incluindo histórico de rodadas de investimento, valores captados, localização e áreas de atuação.
O objetivo é prever se uma startup terá sucesso (ativa/adquirida) ou insucesso (fechada) com base nessas variáveis.

A base foi adaptada para fins acadêmicos: identificadores, colunas que poderiam gerar vazamento e valores inconsistentes foram removidos. Alguns campos podem conter valores ausentes (NaN), refletindo casos em que o evento não ocorreu ou não foi registrado.

Mais do que buscar o melhor desempenho, este desafio incentiva os participantes a explorar técnicas de pré-processamento, seleção de variáveis e modelagem preditiva aplicadas ao empreendedorismo e inovação.

Visão geral
Tarefa: Classificação binária – prever labels (sucesso/insucesso da startup).
Linhas: 923
Colunas: 32
Observações gerais:
Colunas age_* podem ter NaN (evento não ocorreu).
category_code é uma variável categórica bruta.
As demais dummies são binárias 0/1.
Variável alvo
labels | int64 | Target | Indicador de sucesso | {0, 1} | 1 = sucesso (ativa/adquirida); 0 = fechada.
1 (sucesso) → 597 startups (~64,7%)
0 (insucesso) → 326 startups (~35,3%)

👉 A base está moderadamente desbalanceada, mas adequada para modelagem preditiva.

Idades relativas (anos desde a fundação até o evento)
Medidas contínuas em anos, com 2 casas decimais. Valores negativos foram tratados; NaN significa que o evento não ocorreu / está indisponível.

age_first_funding_year | float64 | Anos até o primeiro funding | ≥ 0 ou NaN.
age_last_funding_year | float64 | Anos até o último funding | ≥ 0 ou NaN.
age_first_milestone_year | float64 | Anos até o primeiro milestone | ≥ 0 ou NaN (muitos NaN).
age_last_milestone_year | float64 | Anos até o último milestone | ≥ 0 ou NaN (muitos NaN).
Estrutura, histórico e escala de captação
relationships | int64 | Contagem de relações (fundadores, executivos, investidores) | ≥ 0.
funding_rounds | int64 | Número de rodadas de captação | ≥ 0.
funding_total_usd | float64 | Total captado (USD) | Outliers suavizados (IQR → valores extremos viraram NaN).
milestones | int64 | Contagem de marcos relevantes | ≥ 0.
avg_participants | float64 | Média de investidores por rodada | ≥ 0.
Localização (dummies de estado – binárias)
Representam o estado onde a startup está sediada. Use como 0/1.
(Substituem state_code, removida para evitar redundância.)

is_CA, is_NY, is_MA, is_TX, is_otherstate | int64 | Estado (Califórnia, Nova Iorque, Massachusetts, Texas, Outros) | {0,1}.
Setor/mercado (categórica + dummies)
category_code | object | Setor principal declarado | string | Requer encoding (One-Hot/Target).
is_software, is_web, is_mobile, is_enterprise, is_advertising, is_gamesvideo, is_ecommerce, is_biotech, is_consulting, is_othercategory | int64 | Indicadores de setor | {0,1}.
Nota: Você pode manter category_code (e fazer o encoding) ou trabalhar apenas com as dummies já disponíveis.

Sinalizadores de financiamento (rodadas e tipos)
has_VC | int64 | Recebeu venture capital? | {0,1}.
has_angel | int64 | Recebeu investimento angel? | {0,1}.
has_roundA, has_roundB, has_roundC, has_roundD | int64 | Teve a respectiva rodada? | {0,1}.
Observações e políticas de dados
Faltantes (NaN): principalmente em age_* e outliers de funding_total_usd → tratar no pipeline (ex.: imputação por mediana ou uso de modelos robustos).
Escalas: variáveis como funding_total_usd, relationships, funding_rounds e avg_participants têm ordens de grandeza diferentes → recomenda-se normalização/padronização (StandardScaler) em modelos lineares.
Vazamento evitado: colunas como status, closed_at, is_top500, datas cruas e identificadores foram removidas.
Balanceamento: verifique a proporção de labels ao treinar; se necessário, use class_weight, threshold tuning ou métricas robustas (AUC/F1).
Este conjunto de dados foi adaptado para fins educacionais e busca promover aprendizado prático em empreendedorismo e modelagem preditiva.

Arquivos
train.csv - dados de treino
test.csv - dados de teste
sample_submission.csv - exemplo de submissão em formato corretoa
