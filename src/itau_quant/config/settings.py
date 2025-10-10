"""Carrega configuracoes globais via Pydantic BaseSettings.

Implementar Settings lendo variaveis de ambiente e arquivos .env; definir
caminhos para data/raw, data/processed, seeds padrao e flags de execucao;
expor um cache singleton Settings() para outros modulos consumirem.
"""
