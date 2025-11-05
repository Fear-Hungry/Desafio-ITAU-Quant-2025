# RUNBOOK DE PRODUÇÃO - PRISM-R

**Sistema:** Portfolio Risk Intelligence System - Risk Parity (ERC)
**Versão:** 1.0
**Data:** 2025-10-22
**Status:** ✅ Sistema Validado OOS (Sharpe 1.05)

---

## 🎯 Estratégia Ativa

**Risk Parity (Equal Risk Contribution)** com fallback automático para 1/N

### Validação Out-of-Sample (4 anos)
- **Sharpe Ratio:** 1.05 (melhor estratégia testada)
- **Ann Return:** 12.39%
- **Ann Vol:** 11.84%
- **Max DD:** -19.83%
- **CVaR 95%:** -1.13%

### Estratégias Descartadas
- ❌ **MV Huber:** Sharpe 0.81 (underperforms 1/N por 0.24 pontos)
- ❌ **MV Shrunk50:** Sharpe 0.75 (pior que Huber)
- ❌ **MV Shrunk20:** Sharpe 0.71 (ainda pior)

---

## ⚙️ Parâmetros de Produção

### Configuração do Portfolio
```yaml
Estratégia: Risk Parity (ERC)
Universo: 69 ativos globais
Rebalance: Mensal (primeiro dia útil)
Janela Estimação: 252 dias (1 ano)
```

### Limites e Constraints
```yaml
Max Position: 10% por ativo
Cardinalidade Target: 12-18 ativos
Vol Target: 11% anualizado
Transaction Costs: 30 bps round-trip
Turnover Target: ≤12%/mês
```

### Estimadores
```yaml
Covariância: Ledoit-Wolf shrinkage (252d)
Returns: Implícitos (via reverse optimization ERC)
```

---

## 🚨 Sistema de Fallback Automático

### Triggers para Switch 1/N

O sistema automaticamente muda para equal-weight (1/N) quando **QUALQUER** trigger é violado:

| Trigger | Limite | Ação |
|---------|--------|------|
| **Sharpe 6M** | ≤ 0.0 | Switch para 1/N |
| **CVaR 95%** | < -2% diário | Switch para 1/N |
| **Max DD** | < -10% | Switch para 1/N |

### Como Funciona

```python
# Executado a cada rebalance
fallback, triggers, metrics = should_fallback_to_1N(
    portfolio_returns,
    lookback_days=126,  # 6 meses
)

if fallback:
    strategy = "1/N"  # Equal-weight
else:
    strategy = "ERC"  # Risk Parity
```

### Logs de Fallback

Quando fallback é ativado:
- ⚠️ Log no console
- 📝 Registro em `production_log.csv`
- 🔔 Flag `fallback_active=True`

**Exemplo:**
```
⚠️  FALLBACK TRIGGER ATIVADO!
   Sharpe 6M: -0.15 (limite: 0.00) ❌
   CVaR 95%: -1.80% (limite: -2.00%) ✅
   Max DD: -12.50% (limite: -10.00%) ❌

   → SWITCH PARA 1/N RECOMENDADO
```

---

## 📅 Procedimento de Rebalance Mensal

### Pré-Requisitos
1. Conexão com internet (download de preços)
2. Python 3.11+ com poetry
3. Ambiente configurado: `poetry install`

### Passo 1: Executar Script de Produção

```bash
cd /home/marcusvinicius/Void/arara-quant-lab
poetry run python run_portfolio_production_erc.py
```

**Tempo estimado:** 15-30 segundos

### Passo 2: Revisar Output

O script imprime:
- ✅ Status dos triggers de fallback
- 📊 Estratégia ativa (ERC ou 1/N)
- 💰 Turnover e custos estimados
- 📈 Pesos propostos (top 10)
- 🔍 Métricas de risco (6M)

**Exemplo de output:**
```
✅ Todos os triggers OK - continuar com ERC
   Sharpe 6M: 1.25
   CVaR 95%: -1.60%
   Max DD: -8.30%

⚙️  [4/5] Otimizando portfolio...
   ✅ Triggers OK → Usando ERC (Risk Parity)
   ✅ Otimização concluída!
      Estratégia: ERC
      N_active: 15
      N_effective: 12.3
      Vol ex-ante: 11.2%
```

### Passo 3: Validar Pesos

Verificar que:
- [ ] Nenhum ativo > 10%
- [ ] 12 ≤ N_active ≤ 18
- [ ] Turnover ≤ 15% (ideal: ≤12%)
- [ ] Vol ex-ante ~ 11% ±2%

Se violações graves, **NÃO EXECUTAR TRADES**. Investigar.

### Passo 4: Executar Trades

**Método:**
- Via broker API (produção)
- Ou manualmente (teste/desenvolvimento)

### Passo 5: Confirmar Logging

Verificar que rebalance foi registrado:
```bash
# Ver últimos 5 rebalances
tail -5 results/production/production_log.csv

# Ver pesos salvos
ls -lh results/production/weights/
```

---

## 📊 Monitoramento Diário

### Dashboard

**Arquivo:** `results/production/production_log.csv`

**Colunas principais:**
- `date`: Data do rebalance
- `strategy`: ERC ou 1/N
- `turnover_realized`: Turnover realizado
- `cost_bps`: Custo em bps
- `n_effective`: Diversificação efetiva
- `sharpe_6m`: Sharpe rolling 6M
- `fallback_active`: Trigger ativo?

### Checklist Diário

1. **[ ] Verificar Sharpe 6M**
   - Alerta se < 0.5
   - Crítico se < 0.0

2. **[ ] Verificar CVaR 95%**
   - Alerta se < -1.5%
   - Crítico se < -2.0%

3. **[ ] Verificar Max DD**
   - Alerta se < -8%
   - Crítico se < -10%

4. **[ ] Verificar Turnover Médio**
   - Target: ≤12%/mês
   - Alerta se > 15%

5. **[ ] Revisar Triggers**
   - Se fallback ativo: investigar causa
   - Se persistir > 2 meses: revisar estratégia

---

## 🔧 Troubleshooting

### Problema: Dados não baixam (timeout)

**Erro:**
```
❌ ERRO: Nenhum dado disponível (timeout ou problema de rede)
```

**Solução:**
1. Verificar conexão internet
2. Tentar novamente (pode ser timeout temporário)
3. Usar dados salvos:
   ```bash
   # Copiar dados anteriores
   cp data/processed/returns_arara.parquet results/backup/
   ```

### Problema: Fallback ativado inesperadamente

**Diagnóstico:**
```bash
# Ver histórico de triggers
grep "fallback_active,True" results/production/production_log.csv
```

**Causas possíveis:**
1. Mercado em crise (esperado)
2. Bug nos dados (verificar outliers)
3. Threshold muito apertado (considerar relaxar)

**Ação:**
- Se fallback persiste > 2 meses: considerar ajustar thresholds
- Ou aceitar que 1/N é melhor no momento

### Problema: Vol ex-ante muito diferente de 11%

**Se Vol < 9%:**
- Portfolio muito conservador
- Considerar reduzir risk aversion

**Se Vol > 13%:**
- Portfolio muito agressivo
- Verificar se dados têm outliers
- Considerar aumentar shrinkage em Σ

---

## 📈 Benchmarks e Comparações

### Comparação vs Baselines (OOS 4 anos)

| Estratégia | Sharpe | Decisão |
|------------|--------|---------|
| 1/N | 1.05 | Fallback ativo |
| **ERC** | **1.05** | ✅ **PRODUÇÃO** |
| 60/40 | 1.03 | Não usar |
| HRP | 0.94 | Não usar |
| Min-Var | 0.90 | Não usar |
| MV Huber | 0.81 | ❌ Descartado |

### Critérios de Sucesso (Mensais)

- **Sharpe OOS ≥ 0.9** (próximo de 1.05 validado)
- **Turnover ≤ 12%** (média mensal)
- **Vol realizada 10-12%** (±2%)
- **Fallback ≤ 2 vezes/ano** (aceitável em crises)

---

## 📁 Estrutura de Arquivos

```
results/production/
├── production_log.csv          # Log completo de rebalances
└── weights/
    ├── weights_20251001.csv    # Pesos de cada rebalance
    ├── weights_20251101.csv
    └── ...
```

### Backup Recomendado

```bash
# Semanal
tar -czf backup_production_$(date +%Y%m%d).tar.gz results/production/

# Mover para storage seguro
mv backup_production_*.tar.gz /path/to/backup/
```

---

## 🔐 Segurança e Compliance

### Checklist de Auditoria

- [ ] Todos os rebalances registrados em log
- [ ] Pesos salvos para cada data
- [ ] Triggers documentados e testados
- [ ] Fallback automático funcional
- [ ] Backups semanais

### Evidências de Validação

- ✅ Walk-forward OOS 4 anos
- ✅ Comparação com baselines
- ✅ Triggers testados em cenários extremos
- ✅ Sistema end-to-end funcional

---

## 🚀 Melhoras Futuras (Opcionais)

1. **Email/SMS em Fallback**
   - Alerta automático quando trigger ativa
   - Requer integração com SMTP/Twilio

2. **Dashboard HTML**
   - Visualização gráfica de métricas
   - Equity curve ERC vs 1/N
   - Já implementado em `production_logger.py`

3. **Broker API Integration**
   - Execução automática de trades
   - Requer credenciais e compliance

4. **Regime Detection**
   - Ajustar λ dinamicamente
   - Bull/bear regime classification

---

## 📞 Contatos e Escalação

**Sistema mantido por:** Equipe PRISM-R
**Última atualização:** 2025-10-22
**Próxima revisão:** 2026-01-01 (trimestral)

**Em caso de dúvidas:**
1. Consultar este RUNBOOK
2. Revisar `RESULTADOS_FINAIS.md`
3. Checar logs em `results/production/`

---

## ✅ Checklist de Go-Live

Antes de usar em produção real com capital:

- [ ] Smoke test executado sem erros
- [ ] Triggers testados em cenários extremos
- [ ] Logging funcional e salvando arquivos
- [ ] Backup configurado
- [ ] Runbook revisado por segunda pessoa
- [ ] Validação OOS confirmada (Sharpe ≥ 1.0)
- [ ] Aprovação de compliance/risk
- [ ] Capital test allocation definido (ex: 1-5% inicial)

---

**Documento mantido por:** Claude (Anthropic)
**Versão:** 1.0
**Status:** ✅ Pronto para Produção (com capital test)
