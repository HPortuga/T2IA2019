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

      for dado in dadosDosFolds:
         dadosAcuracia.append(dado[0])
         dadosLogLoss.append(dado[1])

      print("")

   def calcularMedias(self):