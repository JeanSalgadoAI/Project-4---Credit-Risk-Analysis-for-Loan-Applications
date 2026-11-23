# Dados e escopo

O código original gera esta base artificialmente. Os valores não descrevem pessoas, empresas ou clientes reais.

O CSV existente foi preservado. A revisão não substitui a amostra por dados de clientes.

Arquivo: `data/financial_data.csv` · 500 linhas · 8 colunas.

| Campo | Tipo no CSV | Valores ausentes | Papel na análise |
|---|---|---:|---|
| client_id | int64 | 0 | Campo descritivo; não usado como atributo |
| age | int64 | 0 | Atributo |
| annual_income | int64 | 0 | Atributo |
| credit_score | int64 | 0 | Atributo |
| num_transactions | int64 | 0 | Atributo |
| savings_balance | int64 | 0 | Atributo |
| loan_amount | int64 | 0 | Atributo |
| default_risk | int64 | 0 | Alvo |

Identificadores são artificiais; os nomes das colunas foram mantidos para facilitar a reprodução.

O alvo foi gerado de forma independente dos atributos. Eventuais diferenças de desempenho neste recorte são variação amostral, não evidência de capacidade preditiva em produção. Não utilizar para decisões de crédito; client_id é apenas identificador sintético e fica fora do modelo.
