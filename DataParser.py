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
   dados.pop()

parseIris()