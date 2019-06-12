from sklearn.impute import SimpleImputer
import numpy as np

def parseFiles(fileList):
   fileData = parseData(fileList)
   fileData = preProcessData(fileData)
   return fileData
   
def parseData(fileList):
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
   return fileData

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
         try:
            label = float(label)
            categorias.append(label)
         except:
            if not(label in mapaCategorias):
               mapaCategorias[label] = indiceDaCategoria
               indiceDaCategoria += 1
            categorias.append(mapaCategorias[label])
   
      fileData[file] = separarDadosDaCategoria(fileData[file]), np.asarray(categorias)

   return fileData
      
def separarDadosDaCategoria(dados):
   for linha in range(len(dados)):
      del dados[linha][-1]

   x = len(dados)
   y = len(dados[0])
   new = np.zeros((x, y))
   for i in range(x):
      for j in range(y):
         if (dados[i][j] == "?"):
            new[i][j] = np.nan
         else:
            new[i][j] = dados[i][j]
            
   imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
   dados = imputer.fit_transform(new)
   
   return dados