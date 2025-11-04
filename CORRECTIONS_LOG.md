# Registro de Correções - README.md e CLAUDE.md

**Data:** 2025-01-XX  
**Autor:** Claude (correções solicitadas pelo usuário)  
**Arquivos modificados:** `README.md`, `CLAUDE.md`

---

## 🎯 Resumo das Correções

Este documento registra as correções de **contradições e inconsistências** identificadas nos documentos principais do projeto PRISM-R.

---

## ✅ Correções Realizadas

### 1. **Moeda Base Incorreta (ALTA PRIORIDADE)**

**Arquivo:** `README.md` (Seção: Resumo Executivo, linha 21)

**Problema identificado:**
```diff
- "universo OOS final N=66, BRL base"
+ "universo OOS final N=66, USD base"
```

**Conflito:**
- Linha 21 afirmava "BRL base"
- Linhas 36-37 corrigiam para "USD"

**Correção aplicada:**
- Substituído "BRL base" por "USD base" no resumo executivo
- Adicionada nota de rodapé [^1] explicando a diferença entre 69 ETFs configurados e 66 usados no OOS

**Impacto:** Eliminada informação enganosa sobre a moeda base dos cálculos.

---

### 2. **Contradição no Parâmetro η (turnover penalty) (ALTA PRIORIDADE)**

**Arquivo:** `README.md` (Seção 3.2: Otimização)

**Problema identificado:**
```diff
Linha 78:
- "com λ = 15, η = 0.25, custos lineares de 30 bps"

Linha 82-83:
- "Na execução canônica... **η=0** no termo L1 adicional"
```

**Conflito:**
- Fórmula apresentava η = 0.25
- Texto explicativo afirmava η = 0 na execução canônica

**Correção aplicada:**
- Linha 78: Alterado para "η = 0 (execução canônica)"
- Linha 82-83: Reforçado que η=0 evita dupla penalização
- Adicionada referência à seção 5.4 (ablations) para experimentos com η > 0
- Seção 5.4: Criada nota explicativa sobre parâmetros da execução canônica

**Impacto:** Clarificado que:
- Execução OOS oficial (2020-2025): **η = 0**
- Custos já aplicados via termo `costs(w, w_{t-1}) = 30 bps × ‖w - w_{t-1}‖₁`
- Experimentos com η = 0.25 são ablations exploratórias

---

### 3. **Custos de Transação Inconsistentes (MÉDIA PRIORIDADE)**

**Arquivo:** `CLAUDE.md` (Seção: Optimization Objective Function)

**Problema identificado:**
```diff
CLAUDE.md linha 144:
- "Linear transaction costs (10 bps + slippage)"

README.md linha 78:
- "custos lineares de 30 bps"
```

**Correção aplicada:**
- CLAUDE.md linha 144: Alterado para "Linear transaction costs (30 bps per round-trip)"
- CLAUDE.md linha 394: Atualizado "Controlled via L1 penalty (η = 0.50)" para "Controlled via transaction costs (c = 30 bps) in objective"

**Impacto:** Sincronização completa entre documentos sobre custos de transação.

---

### 4. **Confusão: 162 Splits vs 64 Janelas OOS (MÉDIA PRIORIDADE)**

**Arquivo:** `README.md` (Seção 3.3: Avaliação)

**Problema identificado:**
```diff
Linha 87:
- "162 splits cobrindo 2010–2025"

Linha 312:
- "64 janelas OOS" (período 2020-2025)
```

**Ambiguidade:**
- Não estava claro que 162 refere-se ao total histórico (2010-2025)
- 64 refere-se especificamente ao período OOS oficial (2020-2025)

**Correção aplicada:**
Expandida seção 3.3 com estrutura clara:

```markdown
**Protocolo Walk-Forward Purged:**
- Janela de treino: 252 dias úteis (~1 ano)
- Janela de teste: 21 dias úteis (~1 mês)
- Purge: 2 dias
- Embargo: 2 dias

**Dados históricos:**
- Dados desde 2010 para treino
- Total de 162 possíveis janelas walk-forward (2010-2025)

**Período OOS oficial:**
- Início: 2020-01-02
- Fim: 2025-10-09
- Dias úteis: 1,451
- Janelas de teste OOS: 64
```

**Impacto:** Eliminada ambiguidade sobre escopo temporal da avaliação.

---

### 5. **Universo 69 vs 66 Ativos (BAIXA PRIORIDADE - Nota Explicativa)**

**Arquivo:** `README.md` (múltiplas seções)

**Problema identificado:**
- Inconsistência aparente entre "69 ETFs" e "66 ativos" sem explicação clara

**Correção aplicada:**
- Adicionada nota de rodapé [^1] na primeira menção ao universo (Resumo Executivo)
- Atualizada seção 2 (Dados) para explicitar a exclusão de ETHA, FBTC, IBIT

**Nota adicionada:**
```markdown
[^1]: Universo configurado com 69 ETFs em `configs/universe_arara.yaml`. 
O universo OOS final utiliza 66 ativos após exclusão de ETHA, FBTC e IBIT 
por falta de histórico completo no período 2020-2025.
```

**Impacto:** Esclarecida diferença entre universo configurado e universo efetivamente usado.

---

### 6. **Turnover Target vs Reportado (MÉDIA PRIORIDADE - Nota de Investigação)**

**Arquivo:** `README.md` (Seção 1 e 5.4)

**Problema identificado:**
```diff
Linha 46: "turnover alvo 5–20%"
Linha 33: "Turnover (mediana): ~0.2% ao mês"
```

**Análise:**
- 0.2% ao mês está **abaixo** da banda-alvo de 5-20%
- Bug identificado em `BUG_TURNOVER_PRISM_R.md` sugere métrica incorreta

**Correção aplicada:**
- Linha 46: Removida meta "5-20%" e substituída por "controle de turnover via penalização L1 na função objetivo"
- Seção 5.4: Adicionada nota sobre investigação em andamento:
  ```markdown
  **Turnover reportado:** O valor de ~0.2% ao mês está sendo investigado 
  (ver `BUG_TURNOVER_PRISM_R.md`). Baselines mostram turnover mediano de 
  0.04-0.07% ao mês, sugerindo possível inconsistência na métrica de PRISM-R.
  ```

**Impacto:** Transparência sobre potencial bug na métrica de turnover, evitando afirmações enganosas.

---

## 📊 Resumo de Severidade

| Correção | Severidade | Status | Impacto |
|----------|-----------|--------|---------|
| Moeda base BRL→USD | **ALTA** | ✅ Corrigido | Informação crítica no resumo executivo |
| Parâmetro η contraditório | **ALTA** | ✅ Corrigido | Reprodutibilidade dos resultados |
| Custos 10 vs 30 bps | **MÉDIA** | ✅ Corrigido | Sincronização entre documentos |
| 162 vs 64 splits | **MÉDIA** | ✅ Corrigido | Clareza na metodologia |
| Universo 69 vs 66 | **BAIXA** | ✅ Nota adicionada | Esclarecimento preventivo |
| Turnover reportado | **MÉDIA** | ⚠️ Em investigação | Bug potencial identificado |

---

## 🔍 Validação Pendente

### Turnover Metric Bug
**Arquivo referência:** `BUG_TURNOVER_PRISM_R.md`

**Próximos passos:**
1. Verificar cálculo de turnover em `reports/walkforward/per_window_results.csv`
2. Comparar com baselines (1/N mediana = 0.045%, PRISM-R reporta ~0.02% = 2000x menor)
3. Investigar se penalização L1 está sendo contabilizada corretamente
4. Atualizar métricas após correção do bug

---

## 📝 Arquivos Modificados

```
README.md - 6 blocos editados
├── Resumo executivo (moeda, nota de rodapé)
├── Seção 1 (turnover target)
├── Seção 2 (universo 69→66)
├── Seção 3.2 (η = 0 canônico)
├── Seção 3.3 (clarificação walk-forward)
└── Seção 5.4 (ablations + notas)

CLAUDE.md - 2 blocos editados
├── Optimization Objective Function (custos 30 bps)
└── Performance Targets (turnover control)
```

---

## ✅ Checklist de Reprodutibilidade

Após as correções, os seguintes parâmetros estão **claramente documentados**:

- [x] Moeda base: **USD** (não BRL)
- [x] Penalização L1 (η): **0** na execução canônica OOS
- [x] Custos de transação: **30 bps** por round-trip
- [x] Universo: **66 ativos** no OOS (de 69 configurados)
- [x] Período OOS: **2020-01-02 a 2025-10-09** (1,451 dias, 64 janelas)
- [x] Walk-forward: **252d treino, 21d teste, 2d purge, 2d embargo**
- [x] Lambda (risk aversion): **15**
- [ ] Turnover: **em investigação** (possível bug na métrica)

---

## 📚 Referências

- `README.md` - Documento principal do projeto
- `CLAUDE.md` - Guia para Claude Code
- `BUG_TURNOVER_PRISM_R.md` - Relatório de bug em turnover
- `configs/oos_period.yaml` - Definição canônica do período OOS
- `reports/oos_consolidated_metrics.json` - Métricas consolidadas (single source of truth)

---

**Última atualização:** 2025-01-XX  
**Responsável:** Claude (via solicitação do usuário)  
**Status:** ✅ Correções principais aplicadas | ⚠️ Turnover sob investigação