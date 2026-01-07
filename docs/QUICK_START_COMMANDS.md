# Quick Start Commands

Este guia fornece os comandos essenciais para trabalhar com o projeto PRISM-R (Carteira ARARA).

## Instalação

```bash
# Clone o repositório
git clone <repository-url>
cd arara-quant-lab

# Instale dependências com Poetry
poetry install

# Ative o ambiente virtual (opcional)
poetry shell
```

## Testes

```bash
# Executar todos os testes
poetry run pytest

# Executar testes específicos
poetry run pytest tests/data/
poetry run pytest tests/estimators/
poetry run pytest tests/optimization/

# Com verbose e cobertura
poetry run pytest -v --cov=src/arara_quant
```

## CLI (Interface de Linha de Comando)

O projeto oferece uma CLI unificada para todos os comandos:

```bash
# Ver ajuda geral
poetry run arara-quant --help

# Ver ajuda de um comando específico
poetry run arara-quant run-example --help
```

### Comandos Principais

#### 1. Configurações
```bash
# Mostrar configurações do sistema
poetry run arara-quant show-settings
poetry run arara-quant show-settings --json
```

#### 2. Exemplos (Demonstração)
```bash
# Executar portfolio ARARA básico
poetry run arara-quant run-example arara

# Executar portfolio ARARA robusto
poetry run arara-quant run-example robust
```

#### 3. Pesquisa e Análise
```bash
# Comparar estratégias baseline (1/N, Minimum Variance, Risk Parity)
poetry run arara-quant compare-baselines

# Comparar estimadores de μ e Σ
poetry run arara-quant compare-estimators

# Grid search de hiperparâmetros (shrinkage)
poetry run arara-quant grid-search

# Testar skill de forecast de μ
poetry run arara-quant test-skill

# Backtest walk-forward com validação temporal
poetry run arara-quant walkforward
```

#### 4. Produção
```bash
# Deploy sistema de produção (ERC v2 - recomendado)
poetry run arara-quant production-deploy --version v2

# Deploy sistema de produção (ERC v1 - básico)
poetry run arara-quant production-deploy --version v1
```

#### 5. Otimização e Backtest
```bash
# Otimização com arquivo de configuração
poetry run arara-quant optimize --config configs/optimizer_example.yaml

# Backtest (dry-run: valida config sem simular)
poetry run arara-quant backtest --config configs/optimizer_example.yaml

# Backtest real + relatório walk-forward
poetry run arara-quant backtest --config configs/optimizer_example.yaml --no-dry-run --wf-report
```

## Executar Scripts Diretamente

Alternativamente, você pode executar os scripts diretamente:

```bash
# Exemplos
python scripts/examples/run_portfolio_arara.py
python scripts/examples/run_portfolio_arara_robust.py

# Pesquisa
python scripts/research/run_baselines_comparison.py
python scripts/research/run_estimator_comparison.py

# Produção
python scripts/production/run_portfolio_production_erc_v2.py
```

## Code Quality

```bash
# Lint com ruff
poetry run ruff check src tests

# Format com black
poetry run black src tests

# Type checking (se implementado)
poetry run mypy src
```

## Documentação

```bash
# Validar links internos da documentação
make docs

# Servir docs (static) em http://localhost:8000
make serve-docs
```

## Estrutura de Diretórios

```
.
├── src/arara_quant/     # Código-fonte principal (package)
├── scripts/            # Scripts executáveis
│   ├── examples/       # Demonstrações
│   ├── research/       # Análises
│   └── production/     # Deploy produção
├── tests/              # Testes unitários e integração
├── configs/            # Arquivos de configuração
├── data/               # Dados (raw, processed, cache)
├── docs/               # Documentação
└── notebooks/          # Jupyter notebooks
```

## Próximos Passos

1. **Tutorial Básico:** Veja `docs/QUICKSTART.md` para um tutorial passo a passo
2. **Documentação Técnica:** Leia `docs/specs/PRD.md` para requisitos detalhados do produto
3. **Guia do projeto:** Consulte `docs/README.md` para o indice de documentacao
4. **Resultados:** Veja `outputs/results/` para analises de desempenho

## Solução de Problemas

### Erro: "Command not found: arara-quant"
```bash
# Reinstale o pacote
poetry install
```

### Erro: "Module not found"
```bash
# Verifique se está no diretório correto
pwd  # Deve ser /path/to/arara-quant-lab

# Reinstale dependências
poetry install --no-cache
```

### Problemas com testes
```bash
# Limpe cache do pytest
rm -rf .pytest_cache __pycache__

# Execute testes com verbose
poetry run pytest -v --tb=short
```

## Suporte

Para dúvidas ou problemas:
1. Consulte a documentação em `docs/`
2. Verifique os exemplos em `scripts/examples/`
3. Leia os testes em `tests/` para referência de uso
