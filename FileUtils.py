import glob

def escreverEmArquivo(outputData, outputFile):
   with open(outputFile, "w") as arquivo:
      for algoritmo in outputData:
         arquivo.write("====%s====\n" % algoritmo)
         arquivo.write("**Desempenho geral do algoritmo**\n")
         arquivo.write("- Media Acuracia: %0.3f\n" % outputData[algoritmo].mediaAcuracia)
         arquivo.write("- Media Logistic Loss: %0.3f\n" % outputData[algoritmo].mediaLogLoss)
         arquivo.write("- Desvio Padrao Acuracia: %0.3f\n" % outputData[algoritmo].desvioAcuracia)
         arquivo.write("- Desvio Padrao Logistic Loss: %0.3f\n" % outputData[algoritmo].desvioLogLoss)
         
         arquivo.write("\n\nCombinacoes de parametros ordenadas por desempenho:\n")
         for param in range(len(outputData[algoritmo].parametros)):
            arquivo.write("- Parametros: %s\n" % str(outputData[algoritmo].parametros[param]))
            numFolds = len(outputData[algoritmo].dadosDosFolds[0])
            arquivo.write("- Numero de Folds: %d\n\n" % numFolds)

            for fold in range(numFolds):
               arquivo.write("*Desempenho fold %d\n" % fold)
               arquivo.write("- Acuracia: %0.3f\n" % outputData[algoritmo].dadosDosFolds[param][fold][0])
               arquivo.write("- Logistic Loss: %0.3f\n" % outputData[algoritmo].dadosDosFolds[param][fold][1])
         
            arquivo.write("\n\n")

         arquivo.write("*Desempenho dos parametros testados*\n")
         for result in outputData[algoritmo].paramResults:
            arquivo.write("= Acuracia: %0.3f; Desvio padrao: %0.3f; Parametros: %s\n"
               % (result["mean"], result["std"], str(result["params"])))

         
         
def abrirDataSets(diretorio):
   fileList = glob.glob(diretorio + "*")
   inputFiles = dict()
   for file in fileList:
      fileName = file[len(diretorio):-4]
      inputFiles[fileName] = file
   return inputFiles