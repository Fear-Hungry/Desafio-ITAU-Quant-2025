"""Blueprint for strategy performance reports.

Objetivo
--------
Construir relatórios consolidando métricas, gráficos e metadados em um formato
consumível (HTML/PDF) para stakeholders internos (research, risk, gestores).

Componentes sugeridos
---------------------
- `build_report_bundle(perf_stats, risk_stats, plots, meta)`
    Normaliza as diferentes peças (DataFrames com métricas, figuras) em uma
    estrutura única que será renderizada.
- `render_html(report_bundle, template_path=None)`
    Usa motor de templates (ex.: Jinja2) para gerar HTML responsivo com seções:
    overview da estratégia, métricas agregadas, gráficos (tearsheet/diagnósticos),
    notas metodológicas e metadados (seed, commit hash, universo, janela).
- `export_pdf(html, output_path)`
    Converte o relatório HTML em PDF (ex.: via WeasyPrint, wkhtmltopdf ou Prince),
    incluindo fallback para cenários sem binários instalados (gera apenas HTML).
- `build_and_export_report(perf, risk, diagnostics, plots, meta, output_dir)`
    Função orquestradora chamada ao final dos backtests para salvar artefatos.

Metadados mínimos
-----------------
- Identificação da estratégia (nome, versão, config utilizada).
- Período de teste (`start`, `end`), horizontes (rebalance, lookback).
- Seed aleatório, hash do commit, parâmetros críticos (lambda, tau, constraints).

Requisitos adicionais
---------------------
- Permitir anexar tabelas suplementares (ex.: top holdings, exposures regionais).
- Garantir que o relatório seja autoexplicativo (legendas, fontes dos dados).
- Fornecer hooks para customização/filtros (ex.: selecionar subset de gráficos).

Testes recomendados
-------------------
- `tests/evaluation/test_report.py` cobrindo:
    * renderização básica sem erros (HTML output contém seções-chave),
    * manuseio de metadados ausentes (mensagens claras),
    * conversão para PDF opcional (mock de backend),
    * integração com objetos reais vindos de `evaluation.stats` e `evaluation.plots`.
"""
