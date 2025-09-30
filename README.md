Dataset Description
(English below)

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
Context
This dataset gathers real information about startups from different sectors, including funding history, amounts raised, location, and industry.
The goal is to predict whether a startup will achieve success (active/acquired) or failure (closed) based on these variables.

The dataset was adapted for academic purposes only: identifiers, leakage-prone fields, and inconsistent values were removed. Some fields may contain missing values (NaN), reflecting cases where the event did not occur or was not recorded.

More than achieving the best performance, this challenge encourages participants to explore preprocessing techniques, feature selection, and predictive modeling applied to entrepreneurship and innovation.

Overview
Task: Binary classification – predict labels (success/failure of the startup).
Rows: 923
Columns: 32
General notes:
age_* columns may contain NaN (event not occurred).
category_code is a raw categorical field.
The other dummies are binary 0/1.
Target Variable
labels | int64 | Target | Success indicator | {0, 1} | 1 = success (active/acquired); 0 = closed.
1 (success) → 597 startups (~64.7%)
0 (failure) → 326 startups (~35.3%)

👉 The dataset is moderately imbalanced, but still suitable for predictive modeling.

Relative Ages (years since founding to event)
Continuous measures in years, with 2 decimal places. Negative values were treated; NaN means the event did not occur / is unavailable.

age_first_funding_year | float64 | Years until first funding | ≥ 0 or NaN.
age_last_funding_year | float64 | Years until last funding | ≥ 0 or NaN.
age_first_milestone_year | float64 | Years until first milestone | ≥ 0 or NaN (many NaN).
age_last_milestone_year | float64 | Years until last milestone | ≥ 0 or NaN (many NaN).
Structure, history, and funding scale
relationships | int64 | Count of relationships (founders, executives, investors) | ≥ 0.
funding_rounds | int64 | Number of funding rounds | ≥ 0.
funding_total_usd | float64 | Total raised (USD) | Outliers were smoothed (IQR → extreme values turned into NaN).
milestones | int64 | Count of relevant milestones | ≥ 0.
avg_participants | float64 | Average number of investors per round | ≥ 0.
Location (state dummies – binary)
Represents the state where the startup is based. Use as 0/1.
(Replaced state_code, removed to avoid redundancy.)

is_CA, is_NY, is_MA, is_TX, is_otherstate | int64 | State (California, New York, Massachusetts, Texas, Others) | {0,1}.
Industry/market (categorical + dummies)
category_code | object | Declared main sector | string | Requires encoding (One-Hot/Target).
is_software, is_web, is_mobile, is_enterprise, is_advertising, is_gamesvideo, is_ecommerce, is_biotech, is_consulting, is_othercategory | int64 | Industry indicators | {0,1}.
Note: You can keep category_code (and encode it) or work only with the already available dummies.

Funding indicators (rounds and types)
has_VC | int64 | Received venture capital? | {0,1}.
has_angel | int64 | Received angel investment? | {0,1}.
has_roundA, has_roundB, has_roundC, has_roundD | int64 | Had the respective funding round? | {0,1}.
Data notes and policies
Missing values (NaN): mostly in age_* and funding_total_usd outliers → handle in the pipeline (e.g., median imputation or robust models).
Scaling: variables like funding_total_usd, relationships, funding_rounds, and avg_participants have different orders of magnitude → normalization/standardization (StandardScaler) is recommended for linear models.
Leakage prevention: columns like status, closed_at, is_top500, raw dates, and identifiers were removed.
Balance: check label distribution during training; if needed, use class_weight, threshold tuning, or robust metrics (AUC/F1).
This dataset was adapted for educational purposes and aims to promote hands-on learning in entrepreneurship and predictive modeling.

Files
train.csv - the training set
test.csv - the test set
sample_submission.csv - a sample submission file in the correct format


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
O modelo deve atingir uma acurácia mínima de 80% para ser considerado bem-sucedido (pontuação total ao final do campeonato). Embora a acurácia seja a métrica principal usada na competição, analise também outras métricas como precisão e recall, para melhor interpretação do desempenho do modelo preditivo treinado.
Documentação e Apresentação dos Resultados (demérito de até 2,0 pts):
A documentação clara e a apresentação dos resultados são importantes. O notebook final deve ser bem organizado, com código limpo, e o raciocínio por trás de cada decisão deve ser explicado de forma objetiva e compreensível em células de texto, sem exageros.
Deliverables
Complete Notebook: A detailed account of the process in a Jupyter notebook, from data exploration to model evaluation. You can develop your notebook on Kaggle or import an external .ipynb file.
CSV File: Submission of the results from the best trained model, following the provided template.
Competition Rules
Participation is individual.
Use your Inteli e-mail in the competition for submission identification.
Allowed language and libraries: Python with Numpy, Pandas, Scikit-Learn.
For visualizations: Matplotlib, Seaborn, and/or Plotly.
The use of other libraries or external datasets beyond those provided is not allowed.
Ranking will be defined by accuracy (the higher, the better). Tiebreakers: submission grade and other metrics like precision, recall.
The first place in each class will receive a prize. The overall best accuracy among all classes will receive a special prize.
Copying code or plagiarism will result in immediate disqualification and a grade of zero, with the possibility of disciplinary sanctions.
Evaluation Criteria
Your work will be assessed based on the following aspects:

Data Cleaning and Handling of Missing Values (up to 0.5 pt):
Clearly demonstrate the data preparation process, including how missing values and outliers that may compromise analysis quality were handled.
Categorical Variable Encoding (up to 0.5 pt):
Apply appropriate encoding techniques to transform categorical variables into formats compatible with predictive models, while preserving relevant information.
Data Exploration and Visualization (up to 2.0 pts):
Conduct a thorough exploratory analysis to identify patterns, correlations, and trends. Use graphs and tables to justify your choice of features and models.
Hypothesis Formulation (up to 1.0 pt):
Develop at least three hypotheses regarding factors that may influence the success of startups (e.g., quantity of employees, age etc).
Feature Selection (up to 1.0 pt):
Justify the choice of the most relevant variables based on exploratory analysis and formulated hypotheses.
Model Construction and Evaluation (up to 2.0 pts):
Develop and evaluate suitable machine learning models. Present metrics such as accuracy, precision, recall, and F1-score.
Hyperparameter Finetuning (up to 1.0 pt):
Demonstrate the process of hyperparameter tuning to optimize model performance. Explain the decisions made.
Minimum Accuracy (up to 2.0 pts):
The model must achieve at least 80% accuracy for the submission to be validated. The analysis should also include other metrics for performance interpretation, like precision and recall, for better interpretation.
Documentation and Presentation (up to 2.0 pts deduction):
The final notebook must be clean, organized, and well-documented, with clear and objective explanations in text cells.
Kaggle Competition Foundational Rules
(Non-editable)

Competition participants must also agree to Kaggle's Foundational Competition Rules. These rules will supersede the competition-specific rules in the event of any conflict.
The following Kaggle Competition Foundational Rules (“ Foundational Rules ”) apply to every competition regardless of whether the Sponsor creates competition-specific rules. Any competition-specific rules provided by the Sponsor are in addition to these rules, and in the case of any conflict or inconsistency, these Foundational Rules control and nullify contrary competition-specific rules.

GENERAL COMPETITION RULES - BINDING AGREEMENT
1. ELIGIBILITY
a. To be eligible to enter the Competition, you must be:

a registered account holder at Kaggle.com;
the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor);
not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, or North Korea; and
not a person or representative of an entity under U.S. export controls or sanctions (see: https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx).
b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

2. SPONSOR AND HOSTING PLATFORM
a. The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at www.kaggle.com/terms in addition to these Rules.

3. COMPETITION PERIOD
a. For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the “Competition Period”). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.

4. COMPETITION ENTRY
a. NO PURCHASE NECESSARY TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize.
b. Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.
c. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible.
d. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

5. INDIVIDUALS AND TEAMS
a. Individual Account. You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website.
b. Teams. If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website.
c. Team Merger. Teams may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules.
d. Private Sharing. No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

6. SUBMISSION CODE REQUIREMENTS
a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition (“Competition Code”). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification.
b. Public Code Sharing. You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code.
c. Use of Open Source. Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

7. DETERMINING WINNERS
a. Each Submission will be scored and ranked by the evaluation metric stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set.
b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner.

8. NOTIFICATION OF WINNERS & DISQUALIFICATION
a. The potential winner(s) will be notified by email.
b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition’s judging criteria.
c. In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant’s compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph.
d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle.
e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded.
f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

9. PRIZES
a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants.
b. All Prizes are subject to Competition Sponsor's review and verification of the Participant’s eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions).
c. A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8.
d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed.
e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above.
f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

10. TAXES
a. ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

11. GENERAL CONDITIONS
a. All federal, state, provincial and local laws and regulations apply.

12. PUBLICITY
a. You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.

13. PRIVACY
a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address (“Personal Information”). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <www.kaggle.com/privacy>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <www.kaggle.com/contact>.
b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

14. WARRANTY, INDEMNITY AND RELEASE
a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law.
b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (b) any misrepresentation made by you in connection with the Competition; (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity.
c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

15. INTERNET
a. Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant’s ability to participate.

16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY
a. If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.

17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT
a. Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

18. DEFINITIONS
a. "Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants.
b. An “Entry” is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition.
c. A “Final Submission” is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard.
d. A “Participant” or “Participant User” is an individual who participates in a competition by entering the competition and making a Submission.
e. The “Private Leaderboard” is a ranked display of Participants’ Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition.
f. The “Public Leaderboard” is a ranked display of Participants’ Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition.
g. A “Sponsor” is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules.
h. A “Submission” is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor.
i. A “Team” is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.
