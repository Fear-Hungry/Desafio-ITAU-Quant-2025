# Estimators

Coleção de estimadores de expectativa, risco e utilitários auxiliares usados na
plataforma `itau_quant`. Os módulos aqui expostos encapsulam transformações
estatísticas reutilizáveis e evitam que a camada de orquestração replique
detalhes numéricos delicados (estabilidade, regularização, controle de
dimensionalidade).

## Estrutura

- **`bl.py`** — Implementação completa do fluxo Black-Litterman, incluindo
  *reverse optimization*, construção de matrizes de views e combinação posterior
  com salvaguardas para PSD e condicionamento.
- **`cov.py`** — Estimadores de covariância tradicionais e robustos (amostral,
  Ledoit-Wolf, shrinkage não linear, Tyler, Student-t) além de utilitários para
  projeção PSD e regularização.
- **`factors.py`** — Toolkit de modelos fatoriais: preparação de dados,
  regressões cross-section e time-series, shrinkage de betas, PCA e reconstrução
  de retornos implícitos.
- **`mu.py`** — Estimadores de retorno esperado (média simples, Huber,
  Student-t, shrinkage bayesiano) e integração opcional com Black-Litterman.
- **`validation.py`** — Ajuda na validação temporal com purging/embargo
  (PurgedKFold) e execução de *walk-forward* reutilizável.

## Convenções e dependências

- Código compatível com Python 3.9+ e segue formatação `black`/`ruff` (88 colunas).
- Principais dependências: `numpy`, `pandas`; rotinas evitam `np.linalg.inv`,
  preferindo decomposições estáveis.
- Todos os módulos trabalham com `pandas.DataFrame/Series` para preservar rótulos
  e facilitar integração com pipelines posteriores.

## Uso rápido

```python
import pandas as pd
from itau_quant.estimators import bl, cov, mu

# Covariância shrinkada
cov_df, _ = cov.ledoit_wolf_shrinkage(returns_df)

# Prior de retornos
mean_prior = mu.mean_return(returns_df)

# Views absolutas
views = [{"type": "absolute", "asset": "ITUB4", "expected_return": 0.12, "confidence": 0.7}]

posterior = bl.black_litterman(
    cov=cov_df,
    pi=mean_prior,
    views=views,
    tau=0.05,
    return_intermediates=True,
)

mu_bl = posterior["mu_bl"]
cov_bl = posterior["cov_bl"]
```

## Testes

- Tests unitários ficam em `tests/estimators/` e espelham os módulos acima.
- Execute a suíte com:

  ```bash
  poetry run pytest tests/estimators
  ```

## Boas práticas ao contribuir

- Prefira funções pequenas com anotações de tipo e manipulação cuidadosa de NaNs.
- Valide dimensões, positividade de variâncias e condicionamento antes de
  expor resultados.
- Adicione casos no arquivo `tests/estimators/test_<módulo>.py` sempre que
  introduzir funcionalidade ou correção numérica.
- Para manipulações que envolvam dados proprietários, documente passos de
  reprodução ao invés de versionar arquivos brutos.
