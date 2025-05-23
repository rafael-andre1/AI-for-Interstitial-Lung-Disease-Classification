# Lista (Reunião 30/04)

Representa a ordem pela qual penso fazer o sugerido.

No dia 05/05:

 - Falta receber confirmação de qual método devo usar em "modelo com a melhor loss validation"
 - Falta teste de "resize before augmentations"
 - Falta implementar ideia de agregação de resultados, bem como escolha de critérios
 - Decidi deixar agregação de resultados por features da última camada convolucional para o final (bastante complexo)

## Fine-Tune da ResNet

 1. Aplicar métodos de verificação de **rede com a melhor validation loss**, que substitui aquela que é exportada para o return. Posso precisar de **medidas de verificação de diferença entre melhoria da loss de treino e validação**, por exemplo, se a loss de treino estiver a baixar muito mais rapidamente do que a loss de validação ou se estiver com valores muito baixos então não pode ser considerada como a melhor. ✅
 2. **Descongelar** grande parte das camadas da ResNet pode permitir atualização de pesos a um nível mais específico ao dataset em questão.
 3. **Adicionar métricas** e melhorar a sua análise, bem como **fazer ROC e AUC para o teste** na fase de performance evaluation (threshold continua a ser feita com o split de validação)✅
 4. Fazer **resize dos tensors antes de aplicar as augemntations** pode implicar alguma perda de qualidade mas deve reduzir consideravelmente o tempo de execução do treino.✅

## Agregação de Resultados

Consiste em combinar os resultados obtidos pela ResNet. Isto pode ser feito tanto a nível de resultados de probabilidades como resultado de features.

### Tipo 1 ✅

A nível de **probabilidades**:

*"Acho que devias explorar os resultados em função do paciente em vez de fazeres só a nível de cada slice. Explorar diferentes técnicas de agregar as previsões de todas as slices de cada paciente para teres uma única previsão por paciente e depois comparar com a label de fibrose desse paciente. Algumas possibilidades: média, máximo, mínimo, majority voting (após binarização da previsão), ..."*

Bastante self explanatory, puxo os resultados tal como já estou a fazer, mas arranjo algum tipo de agregação que me permita afirmar se o paciente exibe sinais de fibrose.

### Tipo 2

A nível de **features**:

*"Baseado também nestas features podes voltar ainda ao ponto 1. Em vez de pegares nas previsões e tentares agregar, pegas nas features e agregas (mais uma vez podes fazer média, máximo, mínimo). Podes inclusive tentar pensar como poderias ter uma rede que receba as features das fatias para treinares a previsão a nível do paciente (aqui o principal desafio é que diferentes pacientes têm diferentes números de fatias e isto não é compatível com um MLP standard)."*

Este já é um pouco diferente. Ao invés de pegar em probabilidades, pego na penúltima camada (contém a matriz para cada uma das slices) e junto aquelas que são referentes ao mesmo paciente. De seguida, efetuo algum método de agregação que, dado as informações, junta e produz algum indicativo (como probabilidades) de que o paciente exibe fibrose.

## Classificação 2.5D

*"Para tentar melhorar os resultados, a classificação 2.5D acho que seria o próximo passo (mas que já obrigaria a alterar pelo menos o dataloader). Em vez de dares apenas uma fatia à rede (agora estás a replicar a mesma fatia 3x penso?) dás 3 fatias adjacentes e a previsão que farias é apenas para a fatia central (as fatias adjacentes dão contexto à rede no input)"*

Consiste em usar 3 fatias consecutivas nas camadas R, G e B da ResNet. É uma forma de fornecer mais informação/conteúdo ao modelo.
Neste caso, é importante verificar que a mesma pipeline de augmentations é aplicada às 3 imagens.
O maior problema é que requer uma grande redefinição dos data loaders.


## Identificação de tipo de fibrose {#fibrosis_types}

*"Explorar a estratificação de diferentes tipos de fibrose. Para isto terias de fazer as previsões da rede para todas as fatias mas em vez de olhares para a previsão em si vais extrair as features da última camada convolucional (isto vai-te dar uma matriz PxN em que P é o número de slices e N é o número de features da última camada). Depois dás isto a um algoritmo de redução de dimensionalidade (tipicamente t-SNE ou UMAP) para poderes explorar o espaço das features em 2D. Isto permite ver como diferentes fibroses são codificadas pelo teu modelo e ver quão bem distingue por exemplo diferentes tamanhos de fibrose, diferentes posições anatómicas, fibrose com ou sem honeycombing. O resultado disto seria [gráficos deste género](https://www.researchgate.net/figure/t-SNE-Embeddings-of-48-Transcripts-Color-coded-for-Patient-Attributes-A-Age-red_fig3_325368256)"*

Mesmo conceito de cima, quanto à extração da penúltima camada. De seguida, tal como referido, aplica-se um algoritmo de redução de dimensionalidade onde podemos visualizar alguns clusters. Após essa divisão, faz-se uma verificação manual para identificar clusters como honeycombing e ect.
O importante para este conceito é que requer um modelo bem elaborado, algo com extração de features muito robusto e capaz de detetar que existe fibrose mas diferenciar casos.

## CT-FM

Após aperfeiçoar a ResNet, posso usar um **foundation model** para termos comparativos. O recomendado foi CT-FM. Disseram-me que o input seria 3D, pelo que ainda terei de investigar como usar com dados 2D. O modelo não requer treino com os meus dados, apenas preciso de os passar e obter features.

A classificação, dado que o modelo não está treinado para detetar fibrose, seria feita da seguinte forma:

 1. Retirar penúltima camada, obtendo vetor/matriz de features relativos a cada slice
 2. Determinar um critério que oriente a comparação entre valores de matrizes/vetores de slices labeled, aquelas que já sabemos que possuem fibrose.
 3. Definir thresold de classificação dado essa comparação: "Sabendo que uma matriz de fibrose possui este tipo de estrutura/valores, esta slice possui/não possui, dado o critério acima definido"


### <span style="color: red">Questão</span>

Neste caso não existe treino, devo pegar em por exemplo 80% do dataset para definir o critério com base nas features e avaliar a performance do mesmo nos restantes 20%?

## Desenvolvimento do Report

### Introdução

 1. Incluir objetivos do estágio em tópicos (fazer só no final para ter tudo certinho).
 2. Falar mais sobre o conceito de fibrose, especialmente dos casos em que agreguei manifestações, e caso faça **<a href="#fibrosis_types" style="color: green; text-decoration: none;">separação do tipo de fibrose</a>**, mencionar os tipos encontrados/separados.
 3. Para referências, posso seguir o método que me indicaram hoje, copiar do google scholar, incluir no ficheiro `.bib` do Overleaf e aplicar a referência em LaTeX.

### Estado da Arte

Ver projetos que citem o meu dataset (MEDGIFT) e que executem classificação (os que fazem a mesma coisa que eu) para eu provar que não fiz algo irrelevante.

### Descrição da Base de Dados

Devo fazer uma explicação mais profunda do tipo de dados, nº de pacientes e scans, o tipo de divisão que fiz nos splits, as manifestações que considerei como fibrose, etc.

### Metodologia

Elaborar metodologia de Redes usadas, treino, avaliação, etc.



