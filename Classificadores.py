from DecisionTree import DecisionTree
from DecisionTreeData import DecisionTreeData
from sklearn.model_selection import StratifiedKFold
import DataParser
import numpy as np
import glob
import statistics
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss


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

      melhoresParams = list()

      possiveisParamsDecisionTree = {
         "criterion": ["gini", "entropy"],
         "splitter": ["best", "random"],
         "min_weight_fraction_leaf": np.arange(0, 0.5),
         "presort": [True, False],
         "max_depth": np.arange(5, 10),
         "min_samples_split": np.arange(2, 5),
         "min_samples_leaf": np.arange(1, 5)
      }

      possiveisParamsKNN ={
         "n_neighbors": np.arange(5, 10),
         "weights": ["uniform", "distance"],
         "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
         "leaf_size": np.arange(29, 34),
         "p": np.arange(1, 6),
         "metric" : ["minkowski"]
      }

      possiveisParamsNaiveBayes = {
         "priors": None,
         "var_smoothing": np.arange(2.718**(-9), 2.718**(2))
      }# GaussinanNB

      possiveisParamsRegressaoLogistica = {
         "penalty": ["l1","l2","elasticnet","none"],
         "dual": [True, False],
         "tol": np.arange(2.718**(-5), 2.718**(-3)),
         "fit_intercept": [True, False],
         "intercept_scaling": np.arange(1, 5),
         "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
         "max_iter": np.arange(90, 110),
         "multi_class": ["ovr", "multinomial", "auto"]
      }

      possiveisParamsMLP = {
         "activation": ["identity", "logistic", "tanh", "relu"],
         "solver": ["lbfgs", "sgd", "adam"],
         "alpha": np.arange(0.0001, 0.005),
         "learning_rate": ["constant", "invscaling", "adaptive"],
         "learning_rate_init": np.arange(0.0001, 0.0005),
         "power_t": np.arange(0.5, 1),
         "max_iter": np.arange(200,205),
         "shuffle": [True, False],
         "tol": np.arange(2.718**(-4), 2.718**(-2)),
         "n_iter_no_change": np.arange(8, 13)
      }
      # 
      # Params para os outros algoritmos de classificacao
      # 

      classificadores = {
         "Decision Tree": (tree.DecisionTreeClassifier(), possiveisParamsDecisionTree, melhoresParams)
      }

      dados = (dadosDeTreino, labelsDeTreino)
      encontrarMelhoresParamsPara(classificadores, dados, n_splits)
      
      outputData[file]["Decision Tree"] = DecisionTreeData()
      
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

         quantidadeDeFolds = 5 * len(outputData[file][algoritmo].dadosDosFolds[0])
         outputData[file][algoritmo].mediaAcuracia = outputData[file][algoritmo].somaAcuracia / quantidadeDeFolds
         outputData[file][algoritmo].mediaLogLoss = outputData[file][algoritmo].somaLogLoss / quantidadeDeFolds
         outputData[file][algoritmo].calcularDesvios()

               


   calcularMedias(outputData)
   escreverDados(outputData, outputFiles)
   