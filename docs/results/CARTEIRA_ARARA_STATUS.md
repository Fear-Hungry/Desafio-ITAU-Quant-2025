# Carteira ARARA – Snapshot de Produção

**Data do rebalance:** 2025-10-26
**Pipeline:** `arara_quant.runners.production.run_portfolio_production_erc_v2`
**Fontes:** `results/production/production_log.csv` (linha 33) e `weights_20251026.csv`

---

## 📊 Métricas Recentes

| Métrica (6M ou último rebalance) | Valor | Limite / Observação |
|----------------------------------|-------|---------------------|
| Sharpe Ratio (6M)                | **3.60** | ≥ 0.80 (conforto) |
| Volatilidade realizada           | **6.79%** | Meta: 12% ± 2% |
| Max Drawdown                     | **-2.52%** | Limite: ≥ -15% |
| CVaR 95%                         | **-0.91%** | Risco de cauda controlado |
| N efetivo                        | **18.07** | Diversificação robusta |
| Turnover (vs. equal-weight)      | **134.29%** | Rebalance realizado |
| Custo de transação estimado      | **20.14 bps** | Considerando 15 bps one-way |
| Estratégia ativa                 | `ERC+CashFloor` | Fallback desativado |

---

## 💰 Alocação Atual

### Top holdings (23 ativos)
| Ticker | Peso | Classe |
|--------|------|--------|
| CASH   | 15.00% | Reserva técnica |
| VGIT   | 3.86% | US Treasury (intermediário) |
| VCSH   | 3.86% | US Corporate Short-Term |
| QUAL   | 3.86% | US Equity (Quality) |
| SPY    | 3.86% | US Equity |
| MTUM   | 3.86% | US Equity (Momentum) |
| SCHD   | 3.86% | US Equity (Dividend) |
| SPLV   | 3.86% | US Equity (Low Vol) |
| VYM    | 3.86% | US Equity (Dividend) |
| VTV    | 3.86% | US Equity (Value) |
| USMV   | 3.86% | US Equity (Low Vol) |
| VUG    | 3.86% | US Equity (Growth) |
| IEI    | 3.86% | US Treasury 3-7y |
| IEF    | 3.86% | US Treasury 7-10y |
| TLT    | 3.86% | US Treasury 20y+ |
| BNDX   | 3.86% | International Bonds |
| SHY    | 3.86% | US Treasury 1-3y |
| TIP    | 3.86% | TIPS (Inflation Protected) |
| UUP    | 3.86% | USD Index (FX Hedge) |
| EFA    | 3.86% | International Equity |
| VGK    | 3.86% | International Equity (Europe) |
| AGG    | 3.86% | US Aggregate Bonds |
| VGSH   | 3.86% | US Treasury (curto prazo) |

---

## 📈 Exposição por Classe

| Classe | Exposição | Envelope / Comentário |
|--------|-----------|-----------------------|
| Cash | **15.00%** | Piso normal: 15%, defensivo: 40% |
| US Equity total | **34.78%** | Janela alvo: 10% – 50% |
| Growth (SPY, QQQ, VUG, MTUM) | **11.59%** | ≥ 5% |
| International Equity | **7.73%** | ≥ 3% |
| Bonds totais | **38.63%** | ≤ 50% |
| └ Treasuries | 27.04% | ≤ 45% |
| └ Corporate | 3.86% | dentro do limite geral |
| Commodities | **0.00%** | ≤ 25% |
| Crypto | **0.00%** | ≤ 12% |

---

## ✅ Verificações de Constraint

- **Soma dos pesos:** 1.0000 (incluindo cash)
- **Max position ex-CASH:** 3.86% (limite: 8%)
- **Cardinalidade:** 23 ativos
- **Cash floor:** 15%
- **Fallback:** desativado

---

## 🔍 Observações Operacionais

✅ Nenhum trigger ativo - sistema operando normalmente

- Volatilidade abaixo do alvo (6.8% < 12%) pode indicar regime defensivo ou alta alocação em cash/bonds

**Última atualização:** 2025-10-26 15:28:22
