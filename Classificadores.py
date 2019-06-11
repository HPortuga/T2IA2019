from DecisionTree import DecisionTree
from sklearn.model_selection import StratifiedKFold
import DataParser
import numpy as np
import glob

def criarArquivosDeDados(fileNames):
   for file in fileNames:
      with open(fileNames.get(file), "w") as file:
         file.write("====Fold: 0\n")

def escreverDados(algoritmo, fileName, index, acuracia, logisticLoss, predicao):
   with open(fileName, "a") as file:
      if (index != 0):
         file.write("====Fold: " + str(index) + "\n")
      file.write("Acuracia: " + "%.2f" % acuracia + "\n")
      file.write("Logistic Loss: " + "%.2f" % logisticLoss + "\n")
      file.write("Conjunto Predito: \n")
      file.write(np.array2string(predicao, precision=2, separator=",", suppress_small=True)+"\n")

def abrirDataSets(diretorio):
   fileList = glob.glob(diretorio + "*")
   inputFiles = dict()
   for file in fileList:
      fileName = file[len(diretorio):-4]
      inputFiles[fileName] = file
   return inputFiles

if __name__ == "__main__":
   inputDir = "./DataSets/Raw/"
   inputFiles = abrirDataSets(inputDir)
   inputFiles = DataParser.parseFiles(inputFiles)

   outputDir = "./DadosColetados/"
   outputFiles = dict()
   for file in inputFiles:
      outputFiles[file] = outputDir + file + ".csv"

   # Cria um arquivo novo para cada dataset toda vez que rodar o programa 
   # criarArquivosDeDados(outputFiles)

   outputData = dict()

   skf = StratifiedKFold(n_splits=10)
   for file in inputFiles:
      index = 0

      outputData[file] = {
         "Decision Tree" : {
            # "index": dict(),
            "Soma Acuracia": 0,
            "Soma Logistic Loss": 0,
            "Media Acuracia": 0,
            "Media Logistic Loss": 0
         }
      }

      X, y = inputFiles[file]
      for train_index, test_index in skf.split(X, y):
         dadosDeTreino, dadosDeTeste = X[train_index], X[test_index]
         labelsDeTreino, labelsDeTeste = y[train_index], y[test_index]
   
         decisionTree = DecisionTree(dadosDeTreino, dadosDeTeste, labelsDeTreino, labelsDeTeste)
         outputData[file]["Decision Tree"][index] = dict()
         outputData[file]["Decision Tree"][index]["Acuracia"] = decisionTree.acuracia
         outputData[file]["Decision Tree"][index]["Logistic Loss"] = decisionTree.logisticLoss

         # Demais algoritmos de classificacao

         index += 1

   for file in outputData:
      for algoritmo in outputData[file]:
         for indice in range(index):
            outputData[file][algoritmo]["Soma Acuracia"] += outputData[file][algoritmo][indice]["Acuracia"]
            outputData[file][algoritmo]["Soma Logistic Loss"] += outputData[file][algoritmo][indice]["Logistic Loss"]
         outputData[file][algoritmo]["Media Acuracia"] = "%.2f" % (outputData[file][algoritmo]["Soma Acuracia"] / index)
         outputData[file][algoritmo]["Media Logistic Loss"] = "%.2f" % (outputData[file][algoritmo]["Soma Logistic Loss"] / index)