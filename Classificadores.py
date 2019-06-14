from sklearn.model_selection import StratifiedKFold
import DataParser
import numpy as np
import glob
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from DataObject import DataObject
import FileUtils

def criarArquivosDeDados(fileNames):
   for file in fileNames:
      with open(fileNames.get(file), "w") as file:
         file.write("====Fold: 0\n")

def abrirDataSets(diretorio):
   fileList = glob.glob(diretorio + "*")
   inputFiles = dict()
   for file in fileList:
      fileName = file[len(diretorio):-4]
      inputFiles[fileName] = file
   return inputFiles

def encontrarMelhoresParamsPara(classificadores, dados, n_splits):
   for nome in classificadores:
      print("Procurando melhores parametros para %s" % nome)
      algoritmo = classificadores[nome][0]
      possiveisParams = classificadores[nome][1]
      valores = dados[0]
      labels = dados[1]

      clf = GridSearchCV(algoritmo, possiveisParams, cv=n_splits)
      clf.fit(valores, labels)

      scores = list()
      means = clf.cv_results_["mean_test_score"]
      stds = clf.cv_results_["std_test_score"]
      params = clf.cv_results_["params"]
      for mean, std, param in zip(means, stds, params):
         scores.append( {'mean':mean, 'std':std, 'params':param} )
      
      newlist = list()
      newlist = sorted(scores, key=lambda k: k['mean'], reverse=True)

      classificadores[nome][2].append(newlist[:5])
      classificadores[nome][3].append(scores)

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
      outputData[file] = dict()

      if file == "Zoo": n_splits = 3
      elif file == "Poker": n_splits = 5
      elif file == "Flags": n_splits = 4
      else: n_splits = 10

      skf = StratifiedKFold(n_splits)
      X, y = inputFiles[file]
      dadosDeTreino, labelsDeTreino = X, y

      possiveisParamsDecisionTree = {
         "criterion": ["gini", "entropy"],
         "splitter": ["best", "random"],
         "min_weight_fraction_leaf": np.arange(0, 0.5),
         "presort": [True, False],
         "max_depth": np.arange(5, 10),
         "min_samples_split": np.arange(2, 5),
         "min_samples_leaf": np.arange(1, 5)
      }

      # 
      # Params para os outros algoritmos de classificacao
      # 

      melhoresParams = list()
      paramResults = list()
      classificadores = {
         "Decision Tree": (
            tree.DecisionTreeClassifier(),
            possiveisParamsDecisionTree,
            melhoresParams,
            paramResults)
      }

      dados = (dadosDeTreino, labelsDeTreino)
      encontrarMelhoresParamsPara(classificadores, dados, n_splits)
      
      outputData[file]["Decision Tree"] = DataObject()
      
      for algoritmo in classificadores:
         params = classificadores[algoritmo][2][0]

         index = 0
         for param in params:
            outputData[file][algoritmo].parametros[index] = param["params"]
            
            dadosDosFolds = list()
            for train_index, test_index in skf.split(X, y):
               dadosDeTreino, dadosDeTeste = X[train_index], X[test_index]
               labelsDeTreino, labelsDeTeste = y[train_index], y[test_index]      

               classificador = classificadores[algoritmo][0].set_params(**param["params"])
               classificador.fit(dadosDeTeste, labelsDeTeste)

               acuracia = classificador.score(dadosDeTeste, labelsDeTeste)
               predictProbability = classificador.predict_proba(dadosDeTeste)
               logisticLoss = log_loss(labelsDeTeste, predictProbability)
               
               dadosDosFolds.append((acuracia, logisticLoss))
               outputData[file][algoritmo].somaAcuracia += acuracia
               outputData[file][algoritmo].somaLogLoss += logisticLoss

            outputData[file][algoritmo].dadosDosFolds.append(dadosDosFolds)
            index += 1

         outputData[file][algoritmo].calcularMedias()
         outputData[file][algoritmo].calcularDesvios()

      FileUtils.escreverEmArquivo(outputData[file], outputFiles[file])

