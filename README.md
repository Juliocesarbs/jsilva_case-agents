# Case Técnico — Router de Queries & Seleção de Tools

A ideia do case é criar um fluxo capaz de decidir quando uma mensagem pode ser resolvida de forma simples e quando precisa seguir para uma etapa mais completa, envolvendo seleção de tools e LLM.

Minha abordagem foi começar simples e ir aumentando a complexidade somente quando os resultados mostravam necessidade.

No final, o fluxo ficou assim:

```text
Query
  |
  v
Router
  |
  +---- FAST_PATH ---> resposta simples
  |
  +---- AGENT -------> busca das tools
                            |
                            v
                         Top-2
                            |
                            v
                         Agent / LLM
```

## Router

PPara o roteamento entre FAST_PATH e AGENT, comparei Logistic Regression, Random Forest e Linear SVM utilizando TF-IDF.

Na validação cruzada com 5 folds, o Linear SVM apresentou o melhor resultado médio (~89%), contra aproximadamente 84% do Random Forest e 81% da Logistic Regression.

Também comparei representações word, char e word + char. Como word e word + char apresentaram desempenho semelhante, mantive word TF-IDF pela menor complexidade.

No conjunto de avaliação fornecido pelo case, o modelo final acertou as 30 queries. Como esse conjunto é pequeno, considero os ~89% da validação cruzada uma referência mais adequada para avaliar o comportamento do Router, enquanto os 100% representam apenas o resultado observado no eval fornecido.

Uma evolução para produção seria utilizar confiança e sinais de cobertura da entrada para direcionar casos incertos a um fallback mais robusto.

No dataset de avaliação do case o resultado final foi:

```text
Accuracy: 100%

FAST_PATH: 10/10
AGENT:     20/20
```

A saída também possui um nível de confiança. Uma evolução que eu testaria em produção seria utilizar essa confiança para criar um fallback: decisões mais claras continuam locais e casos de baixa confiança podem ser enviados para um modelo pequeno.

## Seleção das tools

Essa foi a parte que mais exigiu experimentação.

O catálogo possui **285 tools**, muitas com nomes e funções bastante parecidos.

Minha primeira abordagem utilizava nome, descrição e categoria da tool. O resultado inicial foi:

```text
Top-2: 15%
```

Analisando os erros, percebi duas coisas importantes.

A primeira foi que usar mais informações nem sempre ajudava. Utilizando apenas o nome das tools, o Top-2 subiu para:

```text
Top-2: 40%
```

A segunda foi que muitas buscas retornavam uma tool muito específica, enquanto o resultado esperado pelo dataset era uma tool mais geral.

Por exemplo, uma query sobre saldo poderia encontrar uma função muito específica de saldo disponível, enquanto o resultado esperado era simplesmente:

```text
consultar_saldo
```

A partir dessa análise adicionei uma etapa simples de reranking, dando preferência para tools menos específicas entre os candidatos encontrados.

O fluxo final ficou:

```text
Query
  |
  v
TF-IDF
  |
  v
Top-10 candidatos
  |
  v
Reranking
  |
  v
Top-2 tools
```

Com isso o resultado chegou a:

```text
Top-2: 70%
```

## O que também testei

Durante o desenvolvimento experimentei outras abordagens.

Testei busca por caracteres, combinação de rankings e também embeddings com `all-MiniLM-L6-v2`.

Os embeddings conseguiram colocar várias tools esperadas dentro do Top-10, mas tiveram desempenho ruim nas primeiras posições.

Como também adicionavam uma nova dependência e mais processamento, preferi não utilizá-los na solução final.

A ideia aqui foi não adicionar complexidade apenas porque a tecnologia é mais sofisticada.

## Resultado final

Executando o pipeline completo:

| Métrica | Resultado |
|---|---:|
| Router — validação cruzada | **~89%** |
| Router — eval fornecido (30 queries) | **100%** |
| Tool Retrieval — Top-2 | **70%** |
| Economia de custo | **77,8%** |
| Economia de latência | **~95%** |

O relatório completo é gerado em:

```text
reports/candidate_report.json
```

### Sobre o Precision@2

O case chama a métrica de `Precision@K`, então mantive esse nome na implementação.

Como a avaliação verifica se a tool esperada apareceu ou não no Top-K, ao analisar meus experimentos tratei essa métrica como **Hit@K**.

### Sobre a latência

Mantive a forma de medição solicitada no case.

PPor isso a economia de aproximadamente 95% representa o resultado dentro desse benchmark. Como os mocks simulam latência variável, o valor pode apresentar pequenas variações entre execuções.

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute os testes:

```bash
pytest candidate_starter/tests -v
```

Execute o case:

```bash
python -m candidate_starter.run_case
```

## Estrutura do projeto

```text
candidate_starter/
├── router.py
├── retrieval.py
├── harness.py
├── run_case.py
└── tests/

experiments/
├── router/
└── retrieval/

reports/
└── candidate_report.json
```

`candidate_starter` contém a solução utilizada no resultado final.

Em `experiments` mantive os testes e abordagens que fui avaliando durante o desenvolvimento. Preferi deixar essa parte separada para que fosse possível acompanhar o caminho até a solução sem carregar código experimental para a implementação principal.

## Próximos passos

Se esse fluxo fosse evoluir para produção, eu seguiria uma abordagem em camadas.

Queries simples continuariam sendo resolvidas localmente. Casos de baixa confiança poderiam passar por um modelo menor, deixando um modelo maior apenas para situações que realmente precisassem de mais raciocínio.

No retrieval, seguiria uma ideia parecida: busca local para gerar poucos candidatos e um modelo menor apenas para ajudar no reranking quando necessário.

A principal ideia que ficou do desenvolvimento foi:

> **não usar mais complexidade do que o problema precisa.**
