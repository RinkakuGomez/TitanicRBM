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
Hidden = 6
Epoch = 1000
Batch size = 50
Num data = 800
"""

class BateriaPruebasRBM7:
    
    def __init__(self, nameCSV):
        
        input_data = self._readCSV(nameCSV)
        
        #Probar con un lr menor
        self.prueba = RBM(num_epoch=100, learning_rate=0.01, batch_size=50, n_visible=3, n_hidden=8, k=1)
        self.input_random = np.random.permutation(input_data)
        
        self.trainingData = self._create_batch(self.input_random[:round(len(self.input_random)*0.7)]) 
        self.testData = self._create_batch(self.input_random[round(len(self.input_random)*0.7):])
        
        print(len(self.trainingData))
        print(len(self.testData))
        
    def pruebaAccuracy(self):
        
        #Comprobamos si el conjunto de training está vacío
        if(len(self.trainingData) == 0):
            
            print('El conjunto de datos de entrenamiento está vacío. El tamaño es: '+ str(len(self.trainingData)))
                
        else:
            print('El conjunto de datos de entrenamiento tiene un tamaño de: '+ str(len(self.trainingData)))
            
            print('Inicio proceso Training')
            
            
            #Inicio bucle épocas
            for epoch in range(1, self.prueba.num_epoch+1):
                                
                print('La época actual es: '+str(epoch))
                                
                j = 0 # batch ctrl
                accuracy_total = 0

                #Inicio bucle batch
                for batch_n in self.trainingData:
                    
                    #Inicalización de los auxiliares de los diferenciales 
                    arr_dW = []
                    arr_dBh = []
                    arr_dBv = []
                                        
                    i = 1 # data ctrl
                    print('batch número: '+str(j))
                    #Inicio bucle datos
                    for data_n in batch_n:                        
                                                
                        #Calculamos los diferenciales de los parámetros.
                        dW, dbh, dbv, v0, ph0, vk, phk = self.prueba.training(data_n)                        
                        
                        #Insertamos los nuevos diferenciales calculados a la lista
                        arr_dW.append(dW)
                        arr_dBh.append(dbh)
                        arr_dBv.append(dbv)                                                                                         
                        
                        i += 1                                                              
                        
                        #print('Precisión v0: {}'.format(v0))
                        #print('Precisión vk: {}'.format(vk))
                        
                        #Inicialización de la máscara 
                        mask = tf.where(tf.less(v0,0.0), x=tf.zeros_like(v0), y=tf.ones_like(v0))
                        #print('Precisión mask: {}'.format(mask))
                        bool_mask = tf.cast(mask, dtype=tf.bool)
                        #print('Precisión bool mask: {}'.format(bool_mask))
                        
                        #Calculo de accuracy
                        acc = tf.where(bool_mask, x=tf.abs(tf.subtract(v0,vk)), y=tf.zeros_like(v0))
                        #print('Precisión acc: {}'.format(acc))
                        n_values = tf.math.reduce_sum(mask)
                        #print('Precisión n_values: {}'.format(n_values))
                        
                        #print('Precisión divide: {}'.format(tf.divide(tf.math.reduce_sum(acc), n_values)))
                              
                        #print('Precisión subtract: {}'.format(tf.subtract(1.0, tf.divide(tf.math.reduce_sum(acc), n_values))))
                        
                        accuracy_total += tf.subtract(1.0, tf.divide(tf.math.reduce_sum(acc), n_values))
                                                
                    j += 1
                                        
                    #Actualizamos los parámetros
                    self.prueba._updateParams(arr_dW, arr_dBh, arr_dBv)
                    
                    accuracy = accuracy_total/(i)
                    print('Precisión total training: {}'.format(accuracy_total))
                    print('Precisión training: {}'.format(accuracy))
                    
                    accuracy_total = 0
                    
            """
            * Considerar emplear lanzadera para agrupar el update y training. 
            """
        return ''
    
        
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
            
        for data in np.random.permutation(dataset[:801]):
            
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

