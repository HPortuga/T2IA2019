import DataParser
from DecisionTree import DecisionTree
from sklearn.model_selection import StratifiedKFold
import numpy as np

def criarArquivosDeDados(fileNames):
   for file in fileNames:
      with open(fileNames.get(file), "w") as file:
         file.write("====Fold: 0\n")

def escreverDados(fileName, index, acuracia, logisticLoss, predicao):
   with open(fileName, "a") as file:
      if (index != 0):
         file.write("====Fold: " + str(index) + "\n")
      file.write("Acuracia: " + "%.2f" % acuracia + "\n")
      file.write("Logistic Loss: " + "%.2f" % logisticLoss + "\n")
      file.write("Conjunto Predito: \n")
      file.write(np.array2string(predicao, precision=2, separator=",", suppress_small=True)+"\n")

if __name__ == "__main__":
   skf = StratifiedKFold(n_splits=10)
   X, y = DataParser.parseIris()
   
   outputDir = "./DadosColetados/"
   fileNames = {
      "Iris": outputDir + "IrisData.txt",
      "BreastCancer": outputDir + "BreastCancerData.txt"
   }

   irisData = {
      "somaAcuracia": 0.0,
      "somaLogisticLoss": 0.0
   }

   # Cria um arquivo novo toda vez que rodar o programa para cada dataset
   criarArquivosDeDados(fileNames)

   index = 0   
   for train_index, test_index in skf.split(X, y):
      dadosDeTreino, dadosDeTeste = X[train_index], X[test_index]
      labelsDeTreino, labelsDoTeste = y[train_index], y[test_index]

      decisionTree = DecisionTree(dadosDeTreino, dadosDeTeste, labelsDeTreino, labelsDoTeste)
      escreverDados(fileNames["Iris"], index, decisionTree.acuracia, decisionTree.logisticLoss, decisionTree.conjuntoPredito)
      irisData["somaAcuracia"] += decisionTree.acuracia
      irisData["somaLogisticLoss"] += decisionTree.logisticLoss
      
      index += 1

   with open(fileNames["Iris"], "a") as IrisFile:
      IrisFile.write("====\n")
      IrisFile.write("Acuracia Media: " + "%.2f" % (irisData["somaAcuracia"]/index) + "\n")
      IrisFile.write("Perda Logistica Media: " + "%.2f" % (irisData["somaLogisticLoss"]/index) + "\n")