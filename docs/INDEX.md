# Documentação do Projeto PRISM-R

**Índice centralizado de toda a documentação técnica e operacional.**

---

## 📋 Documentos Principais (Raiz do Projeto)

### Core Documentation
- [`README.md`](../README.md) - **Documentação principal completa** (1,547 linhas)
  - Resumo executivo
  - Dados e fontes
  - Universo e regras de constraints
  - Metodologia detalhada (estimadores, otimização, solver)
  - Avaliação (métricas OOS, walk-forward)
  - Resultados e tabelas comparativas
  
- [`CLAUDE.md`](../CLAUDE.md) - **Guia para Claude Code** (desenvolvimento)
  - Arquitetura do projeto
  - Padrões de código
  - Convenções de testes
  - Comandos essenciais

- [`PRD.md`](../PRD.md) - **Product Requirements Document**
  - Especificações técnicas
  - Requisitos funcionais
  - Cronograma e milestones

---

## 📚 Documentação Técnica (docs/)

### Correções e Melhorias
- [`CORRECTIONS_LOG.md`](./CORRECTIONS_LOG.md) - **Log de correções do README.md** (252 linhas)
  - 6 correções críticas aplicadas
  - Moeda base (BRL→USD)
  - Parâmetro η (0.25→0)
  - Custos de transação (10→30 bps)
  - Splits walk-forward (162 vs 64)
  
- [`README_IMPROVEMENTS_SUMMARY.md`](./README_IMPROVEMENTS_SUMMARY.md) - **Sumário de melhorias** (375 linhas)
  - Estatísticas das mudanças (+72% de conteúdo)
  - Expansões técnicas detalhadas
  - Fórmulas e código adicionados
  - Benefícios alcançados

### Validação e Bugs
- [`VALIDATION_CHECKLIST.md`](./VALIDATION_CHECKLIST.md) - **Checklist de validação**
  - Correções críticas
  - Expansões técnicas
  - Rastreabilidade de artefatos
  - Reprodutibilidade
  - Consistência numérica

- [`BUG_TURNOVER_PRISM_R.md`](./BUG_TURNOVER_PRISM_R.md) - **Documentação de bug conhecido**
  - Descrição do problema (turnover 2000x menor que esperado)
  - Impacto nas métricas
  - Status da investigação

- [`VALIDATION_SUMMARY.md`](./VALIDATION_SUMMARY.md) - **Sumário de validação geral**
  - Testes de backtest
  - Validação de constraints
  - Robustez de estimadores
  - Stress tests

### Cobertura e Testes
- [`COVERAGE.md`](./COVERAGE.md) - **Relatório de cobertura de testes**
  - Estatísticas por módulo
  - Áreas com baixa cobertura
  - Recomendações

### Mudanças e Histórico
- [`CHANGELOG.md`](./CHANGELOG.md) - **Histórico de versões**
  - Releases
  - Features adicionadas
  - Bug fixes
  - Breaking changes

---

## 🚀 Guias de Início Rápido

- [`QUICKSTART.md`](./QUICKSTART.md) - **Guia básico de início**
  - Instalação
  - Primeiro backtest
  - Comandos essenciais

- [`QUICKSTART_ROBUSTO.md`](./QUICKSTART_ROBUSTO.md) - **Guia com configurações robustas**
  - Setup para produção
  - Configurações avançadas
  - Troubleshooting

- [`QUICK_START_COMMANDS.md`](./QUICK_START_COMMANDS.md) - **Comandos prontos para uso**
  - Pipeline de dados
  - Backtests
  - Otimização
  - Geração de relatórios

---

## 🔧 Operação e Monitoramento

- [`MONITORING_CHECKLIST.md`](./MONITORING_CHECKLIST.md) - **Checklist de monitoramento**
  - Validação diária
  - Triggers de fallback
  - Alertas de risco
  - Logs e auditoria

- [`ORCHESTRATION_GUIDE.md`](./ORCHESTRATION_GUIDE.md) - **Guia de orquestração**
  - Pipeline completo
  - Scheduling
  - Error handling
  - Deployment

---

## 👥 Desenvolvimento e Contribuição

- [`AGENTS.md`](./AGENTS.md) - **Guia de agentes e automação**
  - Agentes de IA disponíveis
  - Workflows automatizados
  - Integração com CI/CD

---

## 📁 Estrutura de Subdiretórios

### `api/`
- Documentação de API
- Endpoints REST
- Schemas de request/response

### `implementation/`
- Detalhes de implementação
- Design decisions
- Arquitetura de módulos

### `notebooks/`
- Jupyter notebooks exploratórios
- Análises ad-hoc
- Protótipos

### `operations/`
- Runbooks operacionais
- Procedures de manutenção
- Incident response

### `report/`
- Templates de relatórios
- Análises OOS
- Tearsheets

### `results/`
- Documentação de resultados
- Benchmarks
- Comparações históricas

---

## 🔍 Navegação Rápida por Tópico

### Para Reproduzir Resultados OOS
1. [`README.md`](../README.md) - Seção "Quickstart"
2. [`VALIDATION_CHECKLIST.md`](./VALIDATION_CHECKLIST.md) - Testes de reprodutibilidade
3. [`QUICK_START_COMMANDS.md`](./QUICK_START_COMMANDS.md) - Comandos completos

### Para Entender a Metodologia
1. [`README.md`](../README.md) - Seção 4 (Metodologia)
2. [`PRD.md`](../PRD.md) - Especificações técnicas
3. [`CLAUDE.md`](../CLAUDE.md) - Padrões de implementação

### Para Validar Resultados
1. [`README.md`](../README.md) - Seção 5 (Avaliação)
2. [`VALIDATION_SUMMARY.md`](./VALIDATION_SUMMARY.md) - Testes completos
3. [`BUG_TURNOVER_PRISM_R.md`](./BUG_TURNOVER_PRISM_R.md) - Bugs conhecidos

### Para Operar em Produção
1. [`MONITORING_CHECKLIST.md`](./MONITORING_CHECKLIST.md) - Monitoramento diário
2. [`ORCHESTRATION_GUIDE.md`](./ORCHESTRATION_GUIDE.md) - Pipeline de produção
3. `operations/` - Runbooks e procedures

### Para Desenvolver
1. [`CLAUDE.md`](../CLAUDE.md) - Guia principal
2. [`AGENTS.md`](./AGENTS.md) - Automação
3. [`COVERAGE.md`](./COVERAGE.md) - Cobertura de testes
4. `implementation/` - Design decisions

---

## 📊 Documentos por Audiência

### **Executivo / Tomador de Decisão**
- [`README.md`](../README.md) - Resumo executivo (seção 0)
- [`PRD.md`](../PRD.md) - Objetivos e metas
- `report/` - Relatórios e tearsheets

### **Analista Quant / Pesquisador**
- [`README.md`](../README.md) - Metodologia completa (seção 4)
- [`VALIDATION_SUMMARY.md`](./VALIDATION_SUMMARY.md) - Validação estatística
- `notebooks/` - Análises exploratórias

### **Engenheiro de Dados**
- [`README.md`](../README.md) - Seção 2 (Dados e fontes)
- [`QUICK_START_COMMANDS.md`](./QUICK_START_COMMANDS.md) - Pipeline de dados
- `api/` - Schemas e endpoints

### **DevOps / SRE**
- [`ORCHESTRATION_GUIDE.md`](./ORCHESTRATION_GUIDE.md) - Deployment
- [`MONITORING_CHECKLIST.md`](./MONITORING_CHECKLIST.md) - Monitoramento
- `operations/` - Runbooks

### **Desenvolvedor Python**
- [`CLAUDE.md`](../CLAUDE.md) - Padrões de código
- [`COVERAGE.md`](./COVERAGE.md) - Testes
- `implementation/` - Arquitetura

---

## 🎯 Próximos Passos

Após ler esta documentação:

1. **Primeiro uso:** Comece por [`QUICKSTART.md`](./QUICKSTART.md)
2. **Reproduzir OOS:** Siga [`VALIDATION_CHECKLIST.md`](./VALIDATION_CHECKLIST.md)
3. **Entender metodologia:** Leia [`README.md`](../README.md) seções 3-5
4. **Desenvolver:** Consulte [`CLAUDE.md`](../CLAUDE.md)
5. **Operar:** Use [`MONITORING_CHECKLIST.md`](./MONITORING_CHECKLIST.md)

---

**Última atualização:** 2025-01-XX  
**Versão da documentação:** 2.0 (pós-correções)  
**Commit:** 4444e7c