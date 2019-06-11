from sklearn import tree
from sklearn.metrics import log_loss

class DecisionTree:
   def __init__(self, dadosDeTreino, dadosDeTeste, labelsDeTreino, labelsDeTeste):
      self.dadosDeTreino = dadosDeTreino
      self.dadosDeTeste = dadosDeTeste
      self.labelsDeTreino = labelsDeTreino
      self.labelsDeTeste = labelsDeTeste
      self.acuracia = 0
      self.logisticLoss = 0
      self.conjuntoPredito = []
      self.predict()

   def predict(self):
      arvore = tree.DecisionTreeClassifier()
      arvore = arvore.fit(self.dadosDeTreino, self.labelsDeTreino)
      self.conjuntoPredito = arvore.predict(self.dadosDeTeste)
      self.acuracia = arvore.score(self.dadosDeTeste, self.labelsDeTeste)
      predictProbability = arvore.predict_proba(self.dadosDeTreino)
      self.logisticLoss = log_loss(self.labelsDeTreino, predictProbability)

      

