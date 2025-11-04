# Documentação - PRISM-R

**Bem-vindo à documentação técnica do projeto PRISM-R (Portfolio Risk Intelligence System).**

---

## 📖 Índice Completo

👉 **Consulte [`INDEX.md`](./INDEX.md) para o índice completo e navegação detalhada.**

O índice contém:
- 📋 Documentos principais (raiz do projeto)
- 📚 Documentação técnica organizada por categoria
- 🚀 Guias de início rápido
- 🔧 Operação e monitoramento
- 👥 Desenvolvimento e contribuição
- 🔍 Navegação por tópico e audiência

---

## ⚡ Acesso Rápido

### Documentos Principais (Raiz)
- [`../README.md`](../README.md) - **Documentação principal completa** (1,547 linhas)
- [`../CLAUDE.md`](../CLAUDE.md) - Guia para desenvolvedores
- [`../PRD.md`](../PRD.md) - Product Requirements Document

### Correções e Validação
- [`CORRECTIONS_LOG.md`](./CORRECTIONS_LOG.md) - Log de correções do README (252 linhas)
- [`VALIDATION_CHECKLIST.md`](./VALIDATION_CHECKLIST.md) - Checklist de validação
- [`BUG_TURNOVER_PRISM_R.md`](./BUG_TURNOVER_PRISM_R.md) - Bugs conhecidos

### Início Rápido
- [`QUICKSTART.md`](./QUICKSTART.md) - Guia básico
- [`QUICK_START_COMMANDS.md`](./QUICK_START_COMMANDS.md) - Comandos prontos

### Operação
- [`MONITORING_CHECKLIST.md`](./MONITORING_CHECKLIST.md) - Monitoramento diário
- [`ORCHESTRATION_GUIDE.md`](./ORCHESTRATION_GUIDE.md) - Pipeline de produção

---

## 🎯 Por Onde Começar?

### Se você é novo no projeto:
1. Leia [`../README.md`](../README.md) - Seção "Resumo Executivo"
2. Execute [`QUICKSTART.md`](./QUICKSTART.md)
3. Consulte [`INDEX.md`](./INDEX.md) para tópicos específicos

### Se quer reproduzir resultados OOS:
1. [`VALIDATION_CHECKLIST.md`](./VALIDATION_CHECKLIST.md)
2. [`QUICK_START_COMMANDS.md`](./QUICK_START_COMMANDS.md)
3. [`../README.md`](../README.md) - Seção 5 (Avaliação)

### Se quer desenvolver:
1. [`../CLAUDE.md`](../CLAUDE.md) - Padrões de código
2. [`COVERAGE.md`](./COVERAGE.md) - Cobertura de testes
3. `implementation/` - Design decisions

### Se quer operar em produção:
1. [`MONITORING_CHECKLIST.md`](./MONITORING_CHECKLIST.md)
2. [`ORCHESTRATION_GUIDE.md`](./ORCHESTRATION_GUIDE.md)
3. `operations/` - Runbooks

---

## 📁 Estrutura desta Pasta

```
docs/
├── README.md                           # Este arquivo (você está aqui)
├── INDEX.md                            # 📋 ÍNDICE COMPLETO (comece por aqui)
│
├── CORRECTIONS_LOG.md                  # Log de correções do README principal
├── README_IMPROVEMENTS_SUMMARY.md      # Sumário de melhorias (375 linhas)
├── VALIDATION_CHECKLIST.md             # Checklist de validação completo
├── BUG_TURNOVER_PRISM_R.md            # Documentação de bugs conhecidos
├── VALIDATION_SUMMARY.md               # Sumário geral de validação
│
├── QUICKSTART.md                       # Guia básico de início
├── QUICKSTART_ROBUSTO.md              # Guia com config robustas
├── QUICK_START_COMMANDS.md            # Comandos prontos para uso
│
├── MONITORING_CHECKLIST.md            # Checklist de monitoramento
├── ORCHESTRATION_GUIDE.md             # Guia de orquestração
├── COVERAGE.md                         # Relatório de cobertura
├── CHANGELOG.md                        # Histórico de versões
├── AGENTS.md                           # Guia de agentes e automação
│
├── technical_notes.md                  # Notas técnicas diversas
├── user_guide.md                       # Guia do usuário
│
└── [subdirs]/
    ├── api/                           # Documentação de API
    ├── implementation/                # Detalhes de implementação
    ├── notebooks/                     # Jupyter notebooks
    ├── operations/                    # Runbooks operacionais
    ├── report/                        # Templates de relatórios
    └── results/                       # Documentação de resultados
```

---

## 🔗 Links Úteis

- **Repositório:** https://github.com/Fear-Hungry/Desafio-ITAU-Quant
- **Issues:** https://github.com/Fear-Hungry/Desafio-ITAU-Quant/issues
- **CI/CD:** https://github.com/Fear-Hungry/Desafio-ITAU-Quant/actions

---

## 📊 Documentação por Audiência

| Audiência | Documentos Recomendados |
|-----------|-------------------------|
| **Executivo** | [`../README.md`](../README.md) (Resumo), [`../PRD.md`](../PRD.md), `report/` |
| **Analista Quant** | [`../README.md`](../README.md) (Metodologia), [`VALIDATION_SUMMARY.md`](./VALIDATION_SUMMARY.md), `notebooks/` |
| **Eng. Dados** | [`../README.md`](../README.md) (Dados), [`QUICK_START_COMMANDS.md`](./QUICK_START_COMMANDS.md), `api/` |
| **DevOps/SRE** | [`ORCHESTRATION_GUIDE.md`](./ORCHESTRATION_GUIDE.md), [`MONITORING_CHECKLIST.md`](./MONITORING_CHECKLIST.md), `operations/` |
| **Dev Python** | [`../CLAUDE.md`](../CLAUDE.md), [`COVERAGE.md`](./COVERAGE.md), `implementation/` |

---

## 💡 Dica

**Para navegação completa e organizada, sempre consulte [`INDEX.md`](./INDEX.md).**

---

**Última atualização:** 2025-01-XX  
**Versão da documentação:** 2.0  
**Commit:** 4444e7c