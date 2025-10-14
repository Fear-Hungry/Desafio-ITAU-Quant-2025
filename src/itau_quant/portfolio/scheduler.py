"""Blueprint for portfolio rebalance scheduling.

Objetivo
--------
Determinar quando a estratégia deve rebalancear, suportando frequências padrão
(mensal/BMS) e gatilhos especiais, além de se integrar com o motor de backtesting.

Componentes sugeridos
---------------------
- `generate_schedule(dates, frequency="monthly", anchor="BMS")`
    Cria lista de datas-alvo usando funções de `data.processing.calendar`.
- `apply_overrides(schedule, manual_overrides)`
    Permite inserir/remover datas especificadas manualmente (ex.: suspender em feriados).
- `respect_trading_halts(schedule, market_calendar)`
    Remove datas sem pregão ou índices indisponíveis.
- `next_rebalance_date(current_date, schedule)`
    Retorna próxima data útil de rebalance.
- `scheduler(config, market_data)`
    Função orquestradora que monta agenda completa, integrando triggers extras.

Considerações
-------------
- Compatível com múltiplas frequências (mensal, semanal, trimestral).
- Permitir janelas de observação antes do rebalance (ex.: cut-off 3 dias antes).
- Registrar decisões em log para auditoria (data gerada, origem, motivo).
- Expor interface que o backtester consuma diretamente (ex.: iterador).

Testes recomendados
-------------------
- `tests/portfolio/test_scheduler.py` cobrindo:
    * geração de BMS/BME a partir de índice de preços sintético,
    * aplicação de overrides (adições/remoções específicas),
    * interação com triggers extraordinários (pular datas após gatilho recente),
    * comportamento em calendários com feriados prolongados.
"""
