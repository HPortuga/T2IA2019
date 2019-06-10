from sklearn.preprocessing import Imputer
import numpy as np

def parseFiles(fileList):
   fileData = dict()
   
   for file in fileList:
      dados = []
      colunas = []

      with open(fileList[file], "r") as arquivo:
         str = ""
         for linha in arquivo:
            for coluna in linha:
               if (coluna != "," and coluna != "\n"):
                  str += coluna
               else:
                  colunas.append(str)
                  str = "" 

            dados.append(colunas)
            colunas = []
         
      fileData[file] = dados

   preProcessData(fileData)

   print("OI")

def preProcessData(fileData):
   for file in fileData:
      indiceDaCategoria = 0
      mapaCategorias = dict()
      categorias = list()
      mapaAttrs = dict()
      indiceAttrs = 0

      for linha in range(len(fileData[file])):
         for coluna in range(len(fileData[file][0]) - 1):
            dado = fileData[file][linha][coluna]
            try:
               dado = float(dado)
               fileData[file][linha][coluna] = dado
            except ValueError:
               if (dado != "?"):
                  if not (dado in mapaAttrs):
                     mapaAttrs[dado] = indiceAttrs
                     indiceAttrs += 1
                  fileData[file][linha][coluna] = mapaAttrs[dado]

         label = fileData[file][linha][-1]
         if not(label in mapaCategorias):
            mapaCategorias[label] = indiceDaCategoria
            indiceDaCategoria += 1
         categorias.append(mapaCategorias[label])
   
      for linha in range(len(fileData[file])):
         del fileData[file][linha][-1]
         fileData[file][linha] = (fileData[file][linha], categorias[linha])

   print("oi")
      
      

   

def parseIris():
   dados = []
   colunas = []
   with open("./DataSets/Raw/iris.data", "r") as arquivo:
      str = ""
      for linha in arquivo:
         for coluna in linha:
            if (coluna != "," and coluna != "\n"):
               str += coluna
            else:
               colunas.append(str)
               str = "" 

         dados.append(colunas)
         colunas = []

   return divideOsDados(dados)

def divideOsDados(dados):
   indiceDaCategoria = 0
   mapaCategorias = dict()
   categorias = list()
   for i in range(len(dados)):
      dado = dados[i]
      if not (dado[-1] in mapaCategorias):
         mapaCategorias[dado[-1]] = indiceDaCategoria
         indiceDaCategoria += 1

      categorias.append(mapaCategorias[dado[4]])
      del dado[-1]

      for j in range(len(dado)):
         dado[j] = float(dado[j])
      
   dados = np.asarray(dados)
   categorias = np.asarray(categorias)

   return dados, categorias