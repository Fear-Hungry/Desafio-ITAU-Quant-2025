# CVaR Convention - Projeto PRISM-R

> **Convenção Definitiva:** Todo CVaR é reportado **anualizado** usando a fórmula `CVaR_anual = CVaR_diário × √252`.

---

## Contexto da Padronização

### Problema Original

O projeto tinha uma inconsistência entre:
- **Targets** (CLAUDE.md, PRD.md): "CVaR (5%) ≤ 8%" → sem especificar horizonte
- **Métricas reportadas** (README.md, código): "CVaR 95% (1 dia) = -1.27%" → escala diária
- **Monitoramento** (production_monitor.py): CVaR diário < -2% como trigger operacional

Esta mistura de horizontes (anual no alvo, diário no observado) dificultava comparações e validações.

### Solução Adotada

**Opção B: Tudo Anualizado**
- Targets e métricas reportadas usam CVaR **anualizado**
- Triggers operacionais mantêm CVaR **diário** (facilita monitoramento intraday) com nota de equivalência
- Consistência com outras métricas (volatilidade, retorno, custos - todos anualizados)

---

## Fórmula de Anualização

```
CVaR_anual = CVaR_diário × √252
```

**Onde:**
- `CVaR_diário`: Expected Shortfall calculado sobre retornos diários (média dos 5% piores dias)
- `252`: Número de dias úteis de trading por ano
- `√252 ≈ 15.87`: Fator de anualização (raiz quadrada para volatilidade)

**Exemplo:**
- CVaR diário = -1.27%
- CVaR anual = -1.27% × 15.87 ≈ **-20.23%**

---

## Referências no Código

### 1. Cálculo (src/itau_quant/evaluation/oos.py)

```python
cvar_daily = float(_cvar(daily_returns, alpha=0.95))
cvar_annual = cvar_daily * np.sqrt(252)  # √252 scaling

metrics = {
    "cvar_95": cvar_daily,        # Daily CVaR for monitoring
    "cvar_95_annual": cvar_annual, # Annualized CVaR for targets
    ...
}
```

### 2. Targets (PRD.md, CLAUDE.md)

```
Target pós-custos:
• CVaR(5%): ≤ 8% a.a. (anualizado √252 × CVaR diário)
```

### 3. Monitoramento Operacional (production_monitor.py)

```python
# Trigger operacional usa CVaR diário (facilita interpretação intraday)
# CVaR 5% < -2% (daily) → equivalente a ~-32% anual
# Nota: Target do projeto é 8% anual, mas trigger operacional é mais conservador
```

**Equivalências para Triggers:**
- `-2.0%` diário → `-31.7%` anual
- `-1.5%` diário → `-23.8%` anual
- `-1.0%` diário → `-15.9%` anual

---

## Tabela de Conversão Rápida

| CVaR Diário | CVaR Anual (√252) | Interpretação                          |
|-------------|-------------------|----------------------------------------|
| -0.5%       | -7.9%             | Muito baixo risco de cauda             |
| -1.0%       | -15.9%            | Risco moderado                         |
| -1.27%      | -20.2%            | **PRISM-R observado (OOS 2020-2025)**  |
| -1.5%       | -23.8%            | Acima do target (8% a.a.)              |
| -2.0%       | -31.7%            | **Trigger operacional de fallback**    |
| -2.5%       | -39.7%            | Risco extremo                          |

---

## Diretrizes de Uso

### Para Relatórios e Publicações
✅ **Use CVaR anualizado**
- "CVaR 95% = -20.2% a.a."
- "Target: CVaR ≤ 8% a.a."

### Para Debugging e Monitoramento Intraday
✅ **Use CVaR diário** (opcional)
- Mais intuitivo para análise de eventos pontuais
- Facilita comparação com VaR diário
- Sempre adicione nota de conversão: "(equiv. -X% anual)"

### Para Triggers Operacionais
✅ **Use CVaR diário com equivalência**
```python
# Trigger: CVaR diário < -2% (equiv. -32% anual)
if cvar_daily < -0.02:
    trigger_fallback()
```

---

## Validação de Métricas

**Arquivo:** `reports/oos_consolidated_metrics.json`

```json
{
  "cvar_95": -0.0127,          // Daily: -1.27%
  "cvar_95_annual": -0.2016,   // Annual: -20.23%
  ...
}
```

**Checklist de Validação:**
- [ ] `cvar_95_annual ≈ cvar_95 × 15.87` (tolerância ±0.01)
- [ ] Valor reportado no README usa `cvar_95_annual`
- [ ] Comparação com target usa escala anualizada
- [ ] Triggers operacionais documentam equivalência anual

---

## Referências Matemáticas

### CVaR (Expected Shortfall)

```
CVaR_α = E[r | r ≤ VaR_α]
```

Onde:
- `α = 0.05` (5% worst tail)
- `r`: retornos diários
- `VaR_α`: Value-at-Risk no percentil 5%

### Anualização de Volatilidade vs Retorno

**Volatilidade (std):** Escala com √T
```
σ_anual = σ_diário × √252
```

**Retorno (média):** Escala com T
```
μ_anual = (1 + μ_diário)^252 - 1  (composição geométrica)
```

**CVaR (volatilidade de cauda):** Escala como volatilidade
```
CVaR_anual = CVaR_diário × √252
```

---

## Histórico de Mudanças

| Data       | Versão | Mudança                                               |
|------------|--------|-------------------------------------------------------|
| 2025-01-XX | 1.0    | Padronização inicial: CVaR anualizado como padrão     |
|            |        | - Updated PRD.md, CLAUDE.md, README.md                |
|            |        | - Added `cvar_95_annual` to oos.py                    |
|            |        | - Documented triggers in production_monitor.py        |

---

## FAQ

**Q: Por que não usar apenas CVaR diário em todo lugar?**  
A: Inconsistência com outras métricas (vol, retorno). Target de "CVaR ≤ 8%" faria sentido apenas se anual (~0.5% diário é muito restritivo).

**Q: Por que triggers operacionais usam CVaR diário?**  
A: Facilita interpretação em tempo real. Um trader vê "-2% no dia" mais facilmente que "-32% anual equivalente".

**Q: A anualização assume distribuição gaussiana?**  
A: Sim, o fator √252 assume retornos i.i.d. Na prática, fat tails e autocorrelação podem violar isso. CVaR empírico (não paramétrico) mitiga parte do problema.

**Q: Como comparar com benchmarks?**  
A: Sempre usar CVaR anualizado. Benchmarks de mercado reportam CVaR em base anual (ex: "CVaR 95% = 12% a.a.").

---

## Checklist de Implementação

- [x] PRD.md atualizado com "CVaR ≤ 8% a.a."
- [x] CLAUDE.md atualizado com "CVaR ≤ 8% annual"
- [x] src/itau_quant/evaluation/oos.py adiciona `cvar_95_annual`
- [x] README.md reporta CVaR anualizado com fórmula
- [x] production_monitor.py documenta equivalência diário ↔ anual
- [x] REPORT_OUTLINE.md usa CVaR anualizado
- [ ] Atualizar scripts de backtesting para logar ambas métricas
- [ ] Atualizar notebooks de análise exploratória
- [ ] Revisar testes unitários de CVaR

---

**Última revisão:** 2025-01-XX  
**Responsável:** PRISM-R Core Team