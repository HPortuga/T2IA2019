import statistics

class DecisionTreeData():
   def __init__(self):
      self.somaAcuracia = 0
      self.desvioAcuracia = 0
      self.mediaAcuracia = 0
      self.somaLogLoss = 0
      self.mediaLogLoss = 0
      self.desvioLogLoss = 0
      self.dadosDosFolds = list()
      self.parametros = {
         0: "",
         1: "",
         2: "",
         3: "",
         4: "",
      }

   def calcularDesvios(self):
      dadosAcuracia = []
      dadosLogLoss = []

      for dado in self.dadosDosFolds:
         for i in range(len(dado)):
            dadosAcuracia.append(dado[i][0])
            dadosLogLoss.append(dado[i][1])

      self.desvioAcuracia = statistics.pstdev(dadosAcuracia)
      self.desvioLogLoss = statistics.pstdev(dadosLogLoss)

   def calcularMedias(self):
      quantidadeDeParametros = len(self.parametros)
      quantidadeDeFolds = len(self.dadosDosFolds[0]) * quantidadeDeParametros

      self.mediaAcuracia = self.somaAcuracia / quantidadeDeFolds
      self.mediaLogLoss = self.somaLogLoss / quantidadeDeFolds