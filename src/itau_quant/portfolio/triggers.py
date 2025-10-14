"""Blueprint for extraordinary rebalance triggers.

Objetivo
--------
Monitorar condições especiais que justificam antecipar rebalance fora da agenda
regular (ex.: drawdowns extremos, mudanças de regime, eventos de risco).

Componentes sugeridos
---------------------
- `drawdown_trigger(nav_series, threshold)`
    Dispara quando drawdown atual excede limite absoluto ou percentual.
- `cvar_trigger(returns, window, alpha, limit)`
    Avalia CVaR rolling estimado vs. limite definido no mandato.
- `volatility_trigger(returns, window, multiplier)`
    Detecta aumento abrupto de volatilidade (ex.: > 2× média histórica).
- `signal_change_trigger(signals, threshold)`
    Identifica mudanças bruscas em sinais de fatores/indicadores proprietários.
- `trigger_engine(state, data, config)`
    Avalia todos os gatilhos configurados, devolvendo flag/justificativa e data
    sugerida para rebalance extra.
- `cooldown_manager(last_trigger_date, cooldown_period)`
    Evita rebalances extraordinários muito frequentes.

Considerações
-------------
- Integrar com logs estruturados (motivo do gatilho, valores observados).
- Garantir que triggers respeitem restrições regulatórias (ex.: máximo X por mês).
- Permitir parametrização via arquivo de config (thresholds, janelas, cooldown).
- Retornar recomendações claras para o `scheduler`/`rebalancer` (ex.: booleano + contexto).

Testes recomendados
-------------------
- `tests/portfolio/test_triggers.py` cobrindo:
    * drawdown trigger com série sintética (atinge limite esperado),
    * CVaR trigger respondendo a mudança de distribuição,
    * cooldown impedindo múltiplos disparos consecutivos,
    * múltiplos gatilhos ativos retornando lista de motivos.
"""
