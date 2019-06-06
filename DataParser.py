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

      data, target = divideOsDados(dados)

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

   return dados, categorias   
      

parseIris()