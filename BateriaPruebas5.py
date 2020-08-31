# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:36:45 2020

@author: Pc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:12:26 2020

@author: Pc
"""
from modelRBM import RBM
import tensorflow as tf
import numpy as np
import csv

"""
LR = 0.01
Visible = 3
Hidden = 8
Epoch = 1000
Batch size = 50
Num data = 500
"""

"""
Necesito meter el valor de sobrevivir o no.
Probar con 3 input tanto con 8 como con 6 hidden, relación numerica 8/3 6/3

"""

class BateriaPruebasRBM5:
    
    def __init__(self, nameCSV):
        
        input_data = self._readCSV(nameCSV)
    
        self.prueba = RBM(num_epoch=1000, learning_rate=0.01, batch_size=50, n_visible=3, n_hidden=6, k=1)
        self.input_random = np.random.permutation(input_data)
        
        self.trainingData = self._create_batch(self.input_random[:round(len(self.input_random)*0.7)]) 
        self.testData = self._create_batch(self.input_random[round(len(self.input_random)*0.7):])
        
        print(len(self.trainingData))
        print(len(self.testData))

    """
    Prueba Hidden y Visible sample única iteracción
    """
    def pruebaGS_Hidden(self):
        
        prob, bernouille = self.prueba._hidden_sample([1.0,0.0])
        
        print('Prueba prob: ')
        print(self.prueba._sess.run(prob))
        print('Prueba bernouille: ')
        print(self.prueba._sess.run(bernouille))
        print('Prueba shape: ')
        print(self.prueba._sess.run(tf.shape(prob)))
        
        prob, bernouille = self.prueba._visible_sample(bernouille)
        
        print('Prueba prob2: ')
        print(self.prueba._sess.run(prob))
        print('Prueba bernouille2: ')
        print(self.prueba._sess.run(bernouille))       
        
    """
    Prueba Gibb Sampling y Compute Gradient única iteracción
    """
    def pruebaGS(self):
        
        v0, ph0, vk, phk, dW, dbh, dbv = self.prueba.training([1.0,1.0,0.0])
                        
        print('Prueba GS 1: ')
        
        print('V0: ')
        print(v0)
        
        print('PH0: ')
        print(self.prueba._sess.run(ph0))
        
        
        print('Prueba GS 2: ')
        
        print('VK: ')
        print(self.prueba._sess.run(vk))
        
        print('PHK: ')
        print(self.prueba._sess.run(phk))
        
        print('Prueba diff: ')
        
        print('dW: ')
        print(self.prueba._sess.run(dW))
                        
        print('dbh: ')
        print(self.prueba._sess.run(dbh))
        
        print('dbv: ')
        print(self.prueba._sess.run(dbv))
        
        
    """
    Prueba Update Params
    """
    def pruebaUpdate(self):        
        
        # print('First Weigths: ')
        # print(self.prueba._sess.run(self.prueba.weights))
        # initialization = tf.compat.v1.global_variables_initializer()
        
        # with tf.compat.v1.Session() as sess:        
        print("Init W, {}".format(self.prueba.weights))
        
        nameFile = 'PruebaRBM_V3H6E1000.txt'
        nameCSV = 'csvPruebaRBM_V3H6E1000.csv'

        file = open(nameFile,'w+')
        csvFile = open(nameCSV, 'w+')
        
        #TO DO dinamizar cabeceras CSV
        headersWeight = ['weight1.1','weight1.2','weight1.3','weight1.4','weight1.5','weight1.6','weight1.7','weight1.8',
                               'weight2.1','weight2.2','weight2.3','weight2.4','weight2.5','weight2.6','weight2.7','weight2.8',
                               'weight3.1','weight3.2','weight3.3','weight3.4','weight3.5','weight3.6','weight3.7','weight3.8']
            
        headersBHidden = ['hiddenBias1','hiddenBias2','hiddenBias3','hiddenBias4','hiddenBias5','hiddenBias6','hiddenBias7','hiddenBias8']
        headersBVisible = ['visibleBias1','visibleBias2','visibleBias3']
        
        headers = headersWeight + headersBHidden + headersBVisible
        
        writerCSV = csv.DictWriter(csvFile, headers)
        
        dictHeaders = {}
        
        for h in headers:
            dictHeaders[h] = h
        
        writerCSV.writerow(dictHeaders)
        
        file.write('Carácteristicas de la red:\n')
        file.write('\t\t + Número de neuronas visibles: '+str(self.prueba.n_visible)+'\n')
        file.write('\t\t + Número de neuronas ocultas: '+str(self.prueba.n_hidden)+'\n')
        file.write('\t\t + Valor del ratio de aprendizaje: '+str(self.prueba.learning_rate)+'\n')
        file.write('\t\t + Número de épocas: '+str(self.prueba.num_epoch)+'\n')
        file.write('\t\t + Tamaño del batch: '+str(self.prueba.batch_size)+'\n')
        file.write('\t\t + Tamaño del dataset: '+str(len(self.input_random))+'\n')
        file.write('\t\t + Tamaño del bloque de entrenamiento: '+str(len(self.trainingData))+'\n')
        file.write('\t\t + Tamaño del bloque de test: '+str(len(self.testData))+'\n\n')
        
        
        file.write('Valores iniciales\n')
        file.write('\t\t - Weight: {}\n'.format(self.prueba.weights))
        file.write('\t\t - Visible Bias: {}\n'.format(self.prueba.visible_bias))
        file.write('\t\t - Hidden Bias: {} \n\n'.format(self.prueba.hidden_bias))
        
        file.write('Parámetros actualizados en cada época: \n')
        
        
            
        if(len(self.trainingData) == 0):

            print('Training size = '+ len(self.trainingData))
                
        else:
            
            for epoch in range(1, self.prueba.num_epoch+1):
                
                aux = []
                #print(str(epoch))
                
                print('Epoca = '+str(epoch))
                
                file.write('Época '+str(epoch)+'\n')
                
                j = 0 # batch ctrl
                
                for batch_n in self.trainingData:
                    arr_dW = []
                    arr_dBh = []
                    arr_dBv = []
                                        
                    i = 1 # data ctrl
                    
                    for data_n in batch_n:                        
    
                        #dW, dbh, dbv = self.prueba.training(data_n)  
                        dW, dbh, dbv, v0, ph0, vk, phk = self.prueba.training(data_n)                        
                        
                        arr_dW.append(dW)
                        arr_dBh.append(dbh)
                        arr_dBv.append(dbv)  
                                                               
                        if(i == len(batch_n) and j == len(self.trainingData)-1):
                            aux = data_n
                            
                        i += 1
                    j += 1
                    self.prueba._updateParams(arr_dW, arr_dBh, arr_dBv)
                    
                    #Intrtoducimos como nueva línea del fichero csv los valores de los pesos y bias actualizados
                    self._createCSV(writerCSV, headersWeight, self.prueba.weights.numpy(), headersBHidden, self.prueba.hidden_bias.numpy(), headersBVisible, self.prueba.visible_bias.numpy())
                    
                file.write('\t\t - Weight: {}\n'.format(self.prueba.weights))
                file.write('\t\t - Visible Bias: {}\n'.format(self.prueba.visible_bias))
                file.write('\t\t - Hidden Bias: {} \n\n'.format(self.prueba.hidden_bias))
                
                #print('aaaa: '+str(aux))
                phv,h_,pvh,v_ = self.prueba.inference(aux) 
                                
                file.write('\t Resultados de la época\n')
                
                file.write('\t\t - Valor de la entrada: '+str(aux)+'\n')   
                
                file.write('\t\t - Probabilidad hidden: {}\n'.format(phv))
                file.write('\t\t - Bernouille hidden: {}\n'.format(h_))
                                    
                file.write('\t\t - Probabilidad visible: {}\n'.format(pvh))
                file.write('\t\t - Bernouille visible: {}\n\n'.format(v_))
                    
                print("Fin epoca W, {}".format(self.prueba.weights))
        print('Fin entreno')
        
        # sess.run(initialization)
     
    def pruebareduceSum(self, v):
        
        x = tf.reduce_sum(v)
        print('Reduce sum: {}'.format(x))
    
    """
    Prueba inferencia
    """
    def pruebaInferencia(self, v):
        
        phv,h_,pvh,v_ = self.prueba.inference(v)
        
        print('Valor inicial v: '+str(v))
        print('Valor phv: '+str(phv))
        
        print('Valor v bernouille: '+str(v_))
        print('Valor pvh: '+str(pvh))
        
    """
    Prueba Test
    
    """    
    def pruebaTest(self):        
        
        if(len(self.test) == 0):

            print('Training size = '+ len(self.trainingData))
        
        else:
            
            for epoch in range(1, self.prueba.num_epoch+1):
                
                print('Epoca = '+str(epoch))
                
                for batch_n in self.testData:
                    
                    for data_n in batch_n:
                        
                        print('Test data')
                        phv,h_,pvh,v_ = self.prueba.inference(data_n)
            
    """
    def pruebaCrossV(self, input_data):
            
        for trainData, testData in 
        
        return 'OK'
    """            
     
    """
    Prueba creación de CSV para análisis estadístico    
    """
    def _prueba_createCSV(self, nameCSV, weights, hiddenBias, visibleBias):
        
        with open(nameCSV, 'w') as csvFile:
                        
            headersWeight = ['weight1.1','weight1.2','weight1.3','weight1.4','weight1.5','weight1.6','weight1.7','weight1.8',
                               'weight2.1','weight2.2','weight2.3','weight2.4','weight2.5','weight2.6','weight2.7','weight2.8',
                               'weight3.1','weight3.2','weight3.3','weight3.4','weight3.5','weight3.6','weight3.7','weight3.8']
            
            headersBHidden = ['hiddenBias1','hiddenBias2','hiddenBias3','hiddenBias4','hiddenBias5','hiddenBias6','hiddenBias7','hiddenBias8']
            headersBVisible = ['visibleBias1','visibleBias2','visibleBias3']
            
            headers = headersWeight + headersBHidden + headersBVisible
            
            inputWriter = csv.DictWriter(csvFile, headers)
            
            dictHeaders = {}
            
            for h in headers:
                dictHeaders[h] = h
            
            inputWriter.writerow(dictHeaders)
            
            dictionary = {}
                        
            auxWeights = []
            auxBHidden = []
            auxBVisible = []
            
            for w in weights:
                for x in w:
                    auxWeights.append(x)
                    
            for h in hiddenBias:
                print('Array'+str(h))
                for x in h:
                    print('Valor'+str(x))
                    auxBHidden.append(x)
                    
            for v in visibleBias:
                for x in v:
                    auxBVisible.append(x)
                                        
            for w, headers in zip(auxWeights, headersWeight):
                dictionary[headers] = w
                
            for h, headers in zip(auxBHidden, headersBHidden):
                dictionary[headers] = h
                
            for v, headers in zip(auxBVisible, headersBVisible):
                dictionary[headers] = v
            
            for row in dictionary:
                print('Peso: '+ str(dictionary[row]))
            
            print('Dict pesos: ' + str(dictionary))
            
            inputWriter.writerow(dictionary)
            
            return 'hola'
           
    """            
    Métodos AUX
    """
    def _createCSV(self, writerCSV, headersWeight, numpyW, headersBHidden, numpyH, headersBVisible, numpyV):
        
        dictionary = {}
                        
        auxWeights = []
        auxBHidden = []
        auxBVisible = []
        
        """
        Convertimos los arrays multidimensionales en arrays de 1 dimension
        """
        for w in numpyW:
            for x in w:
                auxWeights.append(x)
                
        for h in numpyH:
            for x in h:
                auxBHidden.append(x)
                
        for v in numpyV:
            for x in v:
                auxBVisible.append(x)
        
        """
        Introducimos los pares key-value en el dictionario
        """                          
        for w, headers in zip(auxWeights, headersWeight):
            dictionary[headers] = w
            
        for h, headers in zip(auxBHidden, headersBHidden):
            dictionary[headers] = h
            
        for v, headers in zip(auxBVisible, headersBVisible):
            dictionary[headers] = v
        
        """
        Añadimos la nueva línea al fichero
        """
        writerCSV.writerow(dictionary)
        
        return 'OK'
    
    def _readCSV(self, namecsv):
        
        with open(namecsv, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            i = 0
            fields = []
            data = []
            survived = []
            
            for row in csvreader:
                
                if(i==0):
                    fields.append(row[0])
                    fields.append(row[1])
                    fields.append(row[2])
                else:
                    if(float(row[1]) > 18):
                        aux = 1
                    else:
                        aux = 0
                        
                    data.append([row[0],aux,row[2]])

                    survived.append([row[2],row[2],row[2],row[2]])
                    
                i += 1
           
            self._survived = np.array(survived, np.float32)
            
            return np.array(data, np.float32)
        
    def _divide_dataSet(self, dataset):
    
        trainingData = []
        testData = []
        
        i = 1
            
        for data in np.random.permutation(dataset[:501]):
            
            if(i == 10):
                testData.append(data)
                i = 1
            elif(i > 7):
                testData.append(data)            
                i= i+1
            else:
                trainingData.append(data)
                i= i+1
        
        return self._create_batch(trainingData), self._create_batch(testData)
    
    def _create_batch(self, dataArr):
        
        arrPpl = []
        arrAux = []
        i = 1
                
        if(len(dataArr) == 0):
            
            print('Array sobre el que aplicar el batch está vació')
            
        else:

            for data in dataArr:
                
                if(i <= len(dataArr)):
                    #print('I = '+str(i))
                    arrAux.append(data)
                    
                    if(i % self.prueba.batch_size == 0) or (i == len(dataArr)):
                        arrPpl.append(arrAux)
                        
                        arrAux = []
                                                                        
                    i += 1

            return np.array(arrPpl)       