from DecisionTree import DecisionTree
from sklearn.model_selection import StratifiedKFold
import DataParser
import numpy as np
import glob
import statistics

def criarArquivosDeDados(fileNames):
   for file in fileNames:
      with open(fileNames.get(file), "w") as file:
         file.write("====Fold: 0\n")

def escreverDados(outputData, outputFiles):
   for file in outputFiles:
      with open(outputFiles[file], "w") as arquivo:
         for algoritmo in outputData[file]:
            arquivo.write("===" + algoritmo + "===\n")
            arquivo.write("Desvios padrao:\n")
            arquivo.write("= Acuracia: " + "%.2f" % (outputData[file][algoritmo]["Desvio Acuracia"]))
            arquivo.write("\n= Logistic Loss: " + "%.2f" % (outputData[file][algoritmo]["Desvio Logistic Loss"]))

            arquivo.write("\n\nMedias:\n")
            arquivo.write("= Acuracia: " + outputData[file][algoritmo]["Media Acuracia"])
            arquivo.write("\n= Logistic Loss: " + outputData[file][algoritmo]["Media Logistic Loss"])

            arquivo.write("\n\nFolds:")
            numFolds = len(outputData[file][algoritmo]) - 6
            for indice in range(numFolds):
               arquivo.write("\n= Fold " + str(indice) + "\n")
               arquivo.write("Acuracia: " + "%.2f" % (outputData[file][algoritmo][indice]["Acuracia"]) + "\n")
               arquivo.write("Logistic Loss: " + "%.2f" % (outputData[file][algoritmo][indice]["Logistic Loss"]) + "\n")

def abrirDataSets(diretorio):
   fileList = glob.glob(diretorio + "*")
   inputFiles = dict()
   for file in fileList:
      fileName = file[len(diretorio):-4]
      inputFiles[fileName] = file
   return inputFiles

def calcularMedias(outputData):
   for file in outputData:
      for algoritmo in outputData[file]:
         dadosAcuracia = []
         dadosLogLoss = []

         numFolds = len(outputData[file][algoritmo]) - 4
         for indice in range(numFolds):
            outputData[file][algoritmo]["Soma Acuracia"] += outputData[file][algoritmo][indice]["Acuracia"]
            outputData[file][algoritmo]["Soma Logistic Loss"] += outputData[file][algoritmo][indice]["Logistic Loss"]
            dadosAcuracia.append(outputData[file][algoritmo][indice]["Acuracia"])
            dadosLogLoss.append(outputData[file][algoritmo][indice]["Logistic Loss"])

         outputData[file][algoritmo]["Desvio Acuracia"] = statistics.pstdev(dadosAcuracia)
         outputData[file][algoritmo]["Desvio Logistic Loss"] = statistics.pstdev(dadosLogLoss)
         outputData[file][algoritmo]["Media Acuracia"] = "%.2f" % (outputData[file][algoritmo]["Soma Acuracia"] / index)
         outputData[file][algoritmo]["Media Logistic Loss"] = "%.6f" % (outputData[file][algoritmo]["Soma Logistic Loss"] / index)

if __name__ == "__main__":
   inputDir = "./DataSets/Raw/"
   inputFiles = abrirDataSets(inputDir)
   inputFiles = DataParser.parseFiles(inputFiles)

   outputDir = "./DadosColetados/"
   outputFiles = dict()
   for file in inputFiles:
      outputFiles[file] = outputDir + file + ".csv"

   outputData = dict()

   for file in inputFiles:
      outputData[file] = {
         "Decision Tree" : {
            "Soma Acuracia": 0,
            "Soma Logistic Loss": 0,
            "Media Acuracia": 0,
            "Media Logistic Loss": 0
         }
      }

      index = 0      
      
      if file == "Zoo": n_splits = 3
      elif file == "Poker": n_splits = 5
      elif file == "Flags": n_splits = 4
      else: n_splits = 10
      
      skf = StratifiedKFold(n_splits)
      X, y = inputFiles[file]
      
      for train_index, test_index in skf.split(X, y):
         dadosDeTreino, dadosDeTeste = X[train_index], X[test_index]
         labelsDeTreino, labelsDeTeste = y[train_index], y[test_index]
         # cvs - lg loss
         decisionTree = DecisionTree(dadosDeTreino, dadosDeTeste, labelsDeTreino, labelsDeTeste)
         outputData[file]["Decision Tree"][index] = dict()
         outputData[file]["Decision Tree"][index]["Acuracia"] = decisionTree.acuracia
         outputData[file]["Decision Tree"][index]["Logistic Loss"] = decisionTree.logisticLoss

         # Demais algoritmos de classificacao

         index += 1

   calcularMedias(outputData)
   escreverDados(outputData, outputFiles)
   