# PRISM-R — Guia rápido (variante robusta)

Este guia foca nos comandos “robustos” para rodar exemplos, comparar
estimadores/baselines e validar via walk-forward.

## 1) Rodar exemplo robusto

Via CLI:

```bash
poetry run arara-quant run-example robust
```

Ou diretamente:

```bash
poetry run python scripts/examples/run_portfolio_arara_robust.py
```

Configuração padrão do exemplo:

- Universo: `configs/universe_arara_robust.yaml`
- Portfolio: `configs/portfolio_arara_robust.yaml`

## 2) Comparar estimadores (μ/Σ)

```bash
poetry run arara-quant compare-estimators
```

## 3) Comparar baselines

```bash
poetry run arara-quant compare-baselines
```

## 4) Validar via walk-forward (OOS)

```bash
poetry run python scripts/research/run_backtest_walkforward.py
```

## 5) Ajustes recomendados

- Parâmetros do exemplo robusto: `configs/portfolio_arara_robust.yaml`
- Configuração completa (otimização + walk-forward): `configs/optimizer_example.yaml`
- Valide os YAMLs: `make validate-configs`

