"""Define metadados do pacote itau_quant e exporta atalhos publicos.

Implementar constantes de versao, carregar Settings padrao e expor funcoes
principais (ex.: run_backtest) que facilitem o consumo externo do pacote.
"""
"""
Pacote itau_quant: Ferramentas para Análise Quantitativa de Portfólios.

Este pacote fornece uma suíte de ferramentas para download e processamento de dados,
otimização de portfólios e execução de backtests de estratégias de investimento.

Funções de alto nível são expostas aqui para facilitar o consumo como biblioteca.
"""

# --- Metadados do Pacote ---
__version__ = "0.1.0"
__author__ = "Itaú Quant Team"
__email__ = "quant@itau-unibanco.com.br"


# --- Configurações Globais ---
# Em um projeto real, carregaríamos aqui as configurações padrão de um arquivo YAML
# ou variáveis de ambiente, tornando o comportamento do pacote consistente.
# Exemplo:
# from .utils.config import load_default_settings
# settings = load_default_settings()


# --- Atalhos Públicos (Fachada) ---
# O objetivo é expor as funcionalidades mais comuns e de alto nível diretamente
# no namespace do pacote, simplificando o uso para quem importa `itau_quant`.

def run_backtest(strategy_config: dict, returns_data: 'pd.DataFrame') -> 'StrategyResult':
    """
    Executa um ciclo completo de backtest para uma dada estratégia e dados de retorno.

    Esta é uma função de conveniência que orquestra as chamadas aos módulos
    internos de backtesting e avaliação, retornando um objeto de resultado padronizado.

    Args:
        strategy_config (dict): Dicionário com a configuração da estratégia.
        returns_data (pd.DataFrame): DataFrame com a matriz de retornos dos ativos.

    Returns:
        StrategyResult: Um objeto estruturado contendo NAV, trades e métricas.
    """
    # Em um projeto real, esta função conteria a lógica de orquestração.
    # A importação é feita aqui dentro para evitar importações circulares e
    # carregar dependências apenas quando a função é chamada.
    
    # Exemplo de implementação:
    # from .backtesting.engine import run_strategy
    # from .evaluation.metrics import calculate_performance_metrics
    # from .typing import StrategyResult
    
    # nav, trades = run_strategy(returns_data, strategy_config)
    # metrics = calculate_performance_metrics(nav)
    
    # return StrategyResult(nav=nav, trades=trades, metrics=metrics)
    
    # Simulação da chamada
    print(f"Executando backtest com a configuração: {strategy_config['name']}")
    
    # A estrutura StrategyResult viria do arquivo typing.py
    class MockStrategyResult:
        def __init__(self, nav, trades, metrics):
            self.nav = nav
            self.trades = trades
            self.metrics = metrics
        def __repr__(self):
            return f"<StrategyResult metrics={self.metrics}>"

    # Retornaria um objeto real `StrategyResult`
    return MockStrategyResult(
        nav={"2024-01-01": 1.0, "2024-01-02": 1.01},
        trades={"asset": ["AAPL"], "action": ["BUY"]},
        metrics={"sharpe_ratio": 1.5, "max_drawdown": -0.1}
    )


# --- Exposição Explícita ---
# A lista `__all__` define quais nomes serão importados quando um usuário
# fizer `from itau_quant import *`. É uma boa prática para controlar o
# namespace público do pacote.
__all__ = [
    "run_backtest",
    "__version__"
]
