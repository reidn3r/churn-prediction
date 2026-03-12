import pandas as pd
from pydantic import BaseModel

class InferenceInput(BaseModel):
  qtdeDiasD14: float
  propAvgQtdeDias: float
  qtdeDiasPrimeiraTransacao: float
  qtdeDiasUltimaTransacao: float
  propAvgQtdeTransacoes: float
  propAvgQtdePontosPos: float
  propAvgSaldoPontos: float
  propAvgMediaTransacoesDias: float
  qtdeDiasD28: float
  saldoPontosD28: float
  qtdePontosPos: float
  qtdeTransacoes: float
  qtdeTransacoesD7: float
  saldoPontos: float
  mediaTransacoesDias: float
  qtdeTransacoesD28: float
  qtdePontosPosD7: float
  qtdePontosPosD28: float
  qtdePresença: float
  qtdeDias: float
  qtdeTransacoesD14: float
  saldoPontosD14: float
  qtdeChatMessage: float
  saldoPontosD7: float

  def to_dataframe(self):
    return pd.DataFrame([self.model_dump()])