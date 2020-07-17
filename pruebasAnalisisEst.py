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
Batch size = 16
Num data = 800
"""

class PruebasEstadistica:
    
    def __init__(self, nameDataset, n_epoch, lr, bs, num_visible, num_hidden, num_k, arr_fieldString):
        
        #nameCSV = 'dataSetTitanic.csv'
        #input_data = self._readCSVdata(nameCSV, arr_fieldString)
        #arr_fieldString = ['Sex','Age','Survived']
        
        #Titanic dataset 1300
        self.nameCSV = nameDataset
        input_data = self._readCSVdata(self.nameCSV, arr_fieldString)
        
        #Titanic dataset 886
        #nameCSV = 'TitanicSex.csv'
        #input_data = self._readCSV(nameCSV)
                
        
        if(len(input_data) > 0):
            #Probar con un lr menor
            self.prueba = RBM(num_epoch=n_epoch, learning_rate=lr, batch_size=bs, n_visible=num_visible, n_hidden=num_hidden, k=num_k)
            self.input_random = np.random.permutation(input_data)
            print(len(self.input_random))
            
            self.arr_StringComb, self.arr_Comb = self._getCombinatorial() # Obtenemos los valores de la combinatoria
            self.dict_dataComb, self.dict_dataDist = self._init_dictData(self.input_random) # Contiene la distribución de los datos y 
                                                                                            # los datos separados por tipo de valores
            
            arr_train = []  
            arr_test = []                                                                   
            
            # Separamos los valores por la combinación e introducimos el 70% 
            # de ellos en el train arr y el otro en el test
            for key, arr_valores in self.dict_dataDist.items():
                for data in arr_valores[:round(len(arr_valores)*0.7)]:
                    arr_train.append(data)             
                
                for data in arr_valores[round(len(arr_valores)*0.7):]:
                    arr_test.append(data)
            
            #self.trainingData = self._create_batch(self.input_random[:round(len(self.input_random)*0.7)]) 
            #self.testData = self._create_batch(self.input_random[round(len(self.input_random)*0.7):]) 
            
            # Contiene la distribución de los datos y los datos separados 
            # por tipo de valores para los data set de train y test
            self.dict_trainComb, self.dict_trainDist = self._init_dictData(arr_train)
            self.dict_testComb, self.dict_testDist = self._init_dictData(arr_test)        
            
            # Dividimos los data set de train y test en batch        
            self.trainingData = self._create_batch(np.random.permutation(arr_train))
            self.testData = self._create_batch(np.random.permutation(arr_test))
        
        else:
            print('El fichero del data set seleccionado no contiene ningún registro')
        
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
    * Método: training_Normal_Zeros
    * Descripción: Prueba entrenamiento W con dist normal y bias zeros
    * Variables: 
    *       - None
    """ 
    def training_Normal_Zeros(self):
        
        if(len(self.trainingData) == 0):
            
            print('El conjunto de datos de entrenamiento está vacío. El tamaño es: '+ str(len(self.trainingData)))
                
        else:
            print('El conjunto de datos de entrenamiento tiene un tamaño de: '+ str(len(self.trainingData)))
            
            print('Inicio proceso Training')

            """
            * Al final de cada época mostramos los datos obtenidos
            """                
            print("Inicio epocas:")
            print("\tW, {}".format(self.prueba.weights))
            print("\tbias visible, {}".format(self.prueba.visible_bias))
            print("\tbias hidden, {}".format(self.prueba.hidden_bias))                
            
            #Inicio bucle épocas
            for epoch in range(1, self.prueba.num_epoch+1):            
                
                print('La época actual es: '+str(epoch))
                
                j = 0 # batch ctrl
                
                #Inicio bucle batch
                for batch_n in self.trainingData:
                    
                    #Inicalización de los auxiliares de los diferenciales 
                    arr_dW = []
                    arr_dBh = []
                    arr_dBv = []
                                        
                    i = 1 # data ctrl
                    #Inicio bucle datos
                    for data_n in batch_n:
                        
                        #Calculamos los diferenciales de los parámetros.
                        dW, dbh, dbv, v0, ph0, vk, phk = self.prueba.training(data_n)                        
                        
                        #Insertamos los nuevos diferenciales calculados a la lista
                        arr_dW.append(dW)
                        arr_dBh.append(dbh)
                        arr_dBv.append(dbv)
                        
                        i += 1
                        
                    j += 1
                                        
                    #Actualizamos los parámetros
                    self.prueba._updateParams(arr_dW, arr_dBh, arr_dBv)
                    
            """
            * Al final de cada época mostramos los datos obtenidos
            """                
            print("Fin epocas "+str(epoch)+": ")
            print("\tW, {}".format(self.prueba.weights))
            print("\tbias visible, {}".format(self.prueba.visible_bias))
            print("\tbias hidden, {}".format(self.prueba.hidden_bias))
                
        return ''
    
    """
    * Método: training_Normal_Zeros
    * Descripción: Prueba entrenamiento W con dist normal y bias zeros
    * Variables: 
    *       - None
    """ 
    def pruebaTest(self, name_CSVPrueba):
        
        
        if((len(self.trainingData) == 0) or (len(self.testData) == 0)):
            
            print('El conjunto de datos de entrenamiento está vacío. El tamaño es: '+ str(len(self.trainingData)))
            print('El conjunto de datos de test está vacío. El tamaño es: '+ str(len(self.testData)))
                
        else:
            with open(name_CSVPrueba, 'w+') as file:  
                file.write('Prueba Multi Entrenamiento\n')
                
                file.write('\tValores de la red:\n')
                file.write('\t\tNum epoch: '+str(self.prueba.num_epoch)+'\t\t LR: '+str(self.prueba.learning_rate)+'\n')
                file.write('\t\tNum visible: '+str(self.prueba.n_visible)+'\t\t Num hidden: '+str(self.prueba.n_hidden)+'\n')
                file.write('\t\tBatch size: '+str(self.prueba.batch_size)+'\t\t K: '+str(self.prueba.k)+'\n')
                
                #print('El conjunto de datos de entrenamiento tiene un tamaño de: '+ str(len(self.trainingData)))
                
                pesos_init = self.prueba.weights
                biasV_init = self.prueba.visible_bias
                biasH_init = self.prueba.hidden_bias            
                
                #print('Inicio proceso Training')#Inicio bucle épocas
                
                probs_dictTest = {}
                probs_dictTrain = {}
                
                for epoch in range(1, self.prueba.num_epoch+1):            
                    
                    #print('La época actual es: '+str(epoch))
                    
                    j = 1 # batch ctrl
                    accuracy_batchTrain = 0
                    print('La época actual es: '+str(epoch))
                    file.write('Calculo precisión train:\n\n')
                    file.write('\tLa época actual es: '+str(epoch)+'\n')
                    
                    #Inicio bucle batch
                    for batch_n in self.trainingData:
                        
                        accuracy_dataTrain = 0
                        #Inicalización de los auxiliares de los diferenciales 
                        arr_dW = []
                        arr_dBh = []
                        arr_dBv = []
                                            
                        i = 1 # data ctrl

                        #Inicio bucle datos
                        for data_n in batch_n:                        
                                                        
                            #Calculamos los diferenciales de los parámetros.
                            dW, dbh, dbv, v0, ph0, vk, phk = self.prueba.training(data_n)                                                
    
                            #Insertamos los nuevos diferenciales calculados a la lista
                            arr_dW.append(dW)
                            arr_dBh.append(dbh)
                            arr_dBv.append(dbv)
                            
                            #Inicialización de la máscara 
                            mask = tf.where(tf.less(v0,0.0), x=tf.zeros_like(v0), y=tf.ones_like(v0))
                            #print('Precisión mask: {}'.format(mask))
                            bool_mask = tf.cast(mask, dtype=tf.bool)
                            #print('Precisión bool mask: {}'.format(bool_mask))
                            
                            #Calculo de accuracy
                            acc = tf.where(bool_mask, x=tf.abs(tf.subtract(v0,vk)), y=tf.zeros_like(v0))
                            #print('Precisión acc: {}'.format(acc))
                            n_values = tf.math.reduce_sum(mask)
                            
                            accuracy_dataTrain += tf.subtract(1.0, tf.divide(tf.math.reduce_sum(acc), n_values))
                        
                            i += 1
                        
                        accuracy_batchTrain += accuracy_dataTrain/i   
                        j += 1
                                            
                        #Actualizamos los parámetros
                        self.prueba._updateParams(arr_dW, arr_dBh, arr_dBv)

                    accuracy_epochTrain = accuracy_batchTrain/j
                    
                    print('Precisión training: {}'.format(accuracy_epochTrain))
                    file.write('\t\tPrecisión training: {}\n'.format(accuracy_epochTrain))
                        
                file.write('\n\n\tValor inicial pesos y bias\n')
                file.write("\t\tW, {}\n".format(pesos_init))
                file.write("\t\tbias visible, {}\n".format(biasV_init))
                file.write("\t\tbias hidden, {}\n".format(biasH_init))
                
                file.write('Valor final pesos y bias\n')
                file.write("\tW, {}\n".format(self.prueba.weights))
                file.write("\tbias visible, {}\n".format(self.prueba.visible_bias))
                file.write("\tbias hidden, {}\n".format(self.prueba.hidden_bias))        
                print('Fin proceso training')
                
                            
                """
                * Comenzamos proceso de test
                """
                
                #Obtenemos las claves String para el diccionario            
                
                for strg in self.arr_StringComb:
                    probs_dictTest[strg] = 0
                    probs_dictTrain[strg] = 0
                
                #print('Inicio proceso test')
                #Inicio bucle batch
                j = 1 # batch ctrl
                file.write('Calculo precisión test:\n\n')
                
                accuracy_batchTest = 0
                
                for batch_n in self.testData:
                    
                    i = 1 # data ctrl
                    accuracy_dataTest = 0
                    
                    #Inicio bucle data
                    for data_n in batch_n:
                        
                        aux_data = data_n
                        aux_data[2] = -1.0
                        
                        phv,h_,pvh,v_ = self.prueba.inference(aux_data)
                        
                        str_vNew = ''
                        
                        for binary in v_[0]:
                            str_vNew += str(int(binary))
                            
                        for strg in self.arr_StringComb:
                            if(str_vNew == strg):
                                probs_dictTest[strg] += 1
                        
                        #Inicialización de la máscara 
                        mask = tf.where(tf.less(data_n,0.0), x=tf.zeros_like(data_n), y=tf.ones_like(data_n))
                        #print('Precisión mask: {}'.format(mask))
                        bool_mask = tf.cast(mask, dtype=tf.bool)
                        #print('Precisión bool mask: {}'.format(bool_mask))
                        
                        #Calculo de accuracy
                        acc = tf.where(bool_mask, x=tf.abs(tf.subtract(data_n,v_)), y=tf.zeros_like(data_n))
                        #print('Precisión acc: {}'.format(acc))
                        n_values = tf.math.reduce_sum(mask)
                        
                        accuracy_dataTest += tf.subtract(1.0, tf.divide(tf.math.reduce_sum(acc), n_values))
                        
                        i += 1
                    
                    accuracy_batchTest += accuracy_dataTest/i        
                    j += 1
                    
                    accuracy_test = accuracy_batchTest/j
                    
                    print('Precisión test: {}'.format(accuracy_test))
                    file.write('\t\tPrecisión test: {}\n'.format(accuracy_test))
                    
                for batch_n in self.trainingData:
                    #Inicio bucle data
                    for data_n in batch_n:
                        
                        #aux_data = data_n
                        #aux_data[2] = -1.0
                        
                        phv,h_,pvh,v_ = self.prueba.inference(data_n)
                        
                        str_vNew = ''
                        
                        for binary in v_[0]:
                            str_vNew += str(int(binary))
                            
                        for strg in self.arr_StringComb:
                            if(str_vNew == strg):
                                probs_dictTrain[strg] += 1
                
                file.write('\n\n\tDistribución combinatoria Data set:\n')               
                for key, prob in self.dict_dataComb.items():
                    file.write('\t\tDict Data set \n\tNúmero '+key+', distr: '+str(prob)+', probs: '+str(prob/len(self.input_random))+'\n')
                
                file.write('\n\n\tDistribución combinatoria Train set:\n')               
                for key, prob in self.dict_trainComb.items():
                    file.write('\t\tDict Data Train \n\tNúmero '+key+', distr: '+str(prob)+', probs: '+str(prob/(len(self.trainingData)*self.prueba.batch_size))+'\n') 
                    
                file.write('\n\n\tDistribución combinatoria Recons Train:\n')               
                for key, prob in probs_dictTrain.items():
                    file.write('\t\tDict Recons Data Train \n\tNúmero '+key+', distr: '+str(prob)+', probs: '+str(prob/(len(self.trainingData)*self.prueba.batch_size))+'\n') 
                
                file.write('\n\n\tDistribución combinatoria Test set:\n')
                for key, prob in self.dict_testComb.items():
                    file.write('\t\tDict Data Test \n\tNúmero '+key+', distr: '+str(prob)+', probs: '+str(prob/(len(self.testData)*self.prueba.batch_size))+'\n')
                    
                file.write('\n\n\tDistribución combinatoria Recons Test:\n')
                for key, prob in probs_dictTest.items():
                    file.write('\t\tDict Recons Data Test \n\tNúmero '+key+', distr: '+str(prob)+', probs: '+str(prob/(len(self.testData)*self.prueba.batch_size))+'\n')
                    
                #print('Fin proceso test')
                
                #print('Valores de las probabilidades')
                
                """
                * Reconstrucción del data set completo
                """
                # for data_n in self.arr_Comb:
                    
                #     phv,h_,pvh,v_ = self.prueba.inference(data_n)
                    
                #     print('Valor de los datos de entrada: '+str(data_n))
                #     print('Valor de la probabilidad de hidden: '+str(phv[0]))
                #     print('Valor de los datos de hidden: '+str(h_[0]))
                #     print('Valor de la probailidad de visible recons: '+str(pvh[0]))
                #     print('Valor de los datos de visible recons: '+str(v_[0]))                    
            
        return ''
    
    
    def _pruebaDict(self):
        
        arr_num = [1,2,5,4,1,2,5,5,4]
        arr_stringNum = []
        probs_dict = {}        
        
        for num in arr_num:
            i = 1
            
            if(len(arr_stringNum) == 0):
                arr_stringNum.append(str(num))
                                    
            else:
    
                for strg in arr_stringNum:
                    
                    if(strg == str(num)):
                        break
                    
                    elif(strg != str(num) and i == len(arr_stringNum)):
                        arr_stringNum.append(str(num))                        
                        break 
                    
                    else:                                        
                        i +=1
                        
        for strg in arr_stringNum:
            probs_dict[strg] = 0
        
        for num in arr_num:
            for strg in arr_stringNum:
                if(strg == str(num)):
                   probs_dict[strg] += 1 
               
                
        for key, prob in probs_dict.items():
            print('Número '+key+', probs: '+str(prob))
            
        return ''
    
    
    def prueba_ClassInput(self):
        
        with open(self.nameCSV, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            
            dictClass = {}
            dictClassSurv = {}
            dictClassMuerte = {}
            
            dictClass['1st'] = 0
            dictClass['2nd'] = 0
            dictClass['3rd'] = 0
            dictClass['crew'] = 0
            
            dictClassSurv['1st'] = 0
            dictClassSurv['2nd'] = 0
            dictClassSurv['3rd'] = 0
            dictClassSurv['crew'] = 0
            
            dictClassMuerte['1st'] = 0
            dictClassMuerte['2nd'] = 0
            dictClassMuerte['3rd'] = 0
            dictClassMuerte['crew'] = 0
            
            for row in csvreader:
                
                dictClass[str(row['PClass'])] += 1 
                
                if(int(row['Survived']) == 0):
                    dictClassMuerte[row['PClass']] += 1
       
                elif(int(row['Survived']) == 1):
                    dictClassSurv[row['PClass']] += 1
                
            for key, prob in dictClass.items():
                print('Clase '+key+', probs: '+str(prob))
            for key, prob in dictClassMuerte.items():
                print('Clase '+key+' ratio muerte: '+str(prob))
            for key, prob in dictClassSurv.items():
                print('Clase '+key+' ratio survi: '+str(prob))
            
                                
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
        
        
        # Introducimos los pares key-value en el dictionario
                                  
        for w, headers in zip(auxWeights, headersWeight):
            dictionary[headers] = w
            
        for h, headers in zip(auxBHidden, headersBHidden):
            dictionary[headers] = h
            
        for v, headers in zip(auxBVisible, headersBVisible):
            dictionary[headers] = v
        
        # Añadimos la nueva línea al fichero
        writerCSV.writerow(dictionary)
        
        return 'OK'
    
    def _readCSV(self, namecsv):
        
        with open(namecsv, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            i = 0
            fields = []
            data = []
            
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
                    
                i += 1
                       
            return np.array(data, np.float32)
       
    def _readCSVdata(self, namecsv, arr_fieldString):
        
        with open(namecsv, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            
            dataSet = []            
            
            for row in csvreader:  
                auxSex = 0
                auxAge = 0
                auxSurvived = 0
                auxClass = 0
                
                for key in arr_fieldString:
                    if(key == 'Age'):
                        if(row[key] == '' or row[key] == ' '):
                            auxAge = 0                         
                        else:
                            if(self.nameCSV == 'Titanic1300.csv'):
                                if(float(row[key]) > 18):
                                    auxAge = 1   
                                else:
                                    auxAge = 0 
                            elif(self.nameCSV == 'Titanic2200.csv'):
                                if(row[key] == 'adult'):
                                    auxAge = 1   
                                else:
                                    auxAge = 0 
                        
                    elif(key == 'Survived'):
                        if(row[key] == '' or row[key] == ' '):
                            auxSurvived = 0
                        else:
                            auxSurvived = row[key]
                    elif(key == 'Sex'):
                        if(row[key] == 'male'):
                            auxSex = 0
                        elif(row[key] == 'female'):
                            auxSex = 1
                        else:
                            auxSex = 0
                    elif(key == 'PClass'):
                        if(row[key] == '1st'):
                            auxClass = 1
                        elif(row[key] == '2nd'):
                            auxClass = 0
                        elif(row[key] == 'crew'):
                            auxClass = 0 #evaluar
                        elif(row[key] == '3rd'):
                            auxClass = 0
                        
                dataSet.append([auxSex,auxAge,auxClass,auxSurvived])
                    
            return np.array(dataSet, np.float32)
        
    def _getCombinatorial(self):
        
        comb = []
        combString = []
        
        if(len(self.input_random) > 0):
            
            for data_n in self.input_random:
               
                
                # Almacenamos en el array las combinaciones de v
                if(len(comb) < (2)**self.prueba.n_visible):
                    
                    key = ''
                    
                    
                    # Si aún no se ha añadido ningún valor al arr de la 
                    # combinatoria se introduce el primero                 
                    if(len(comb) == 0):
                        
                        comb.append(data_n)
                        
                        for binary in data_n:
                            key += str(int(binary))
                            
                        combString.append(key)
                    else:
                        
                        i = 1
                        
                        
                        # Recorremos el arr de la combinatoria                    
                        for v in comb:                                                                        
                            
                            # Comprobamos si el dato de entrada coincide con 
                            # los que contiene el arr de combinatoria, si coincide finalizamos el bucle
                            if(np.array_equal(data_n, v) == True):
                                break
                            
                            # Si el último elemento del arr de combinatoria no 
                            # coincide, se inserta el valor de la entrada
                            elif(np.array_equal(data_n, v) == False and i == len(comb)):                            
                                
                                comb.append(data_n)
                                
                                for binary in data_n:
                                    key += str(int(binary))
                                
                                combString.append(key)
                                
                                break
                            
                            else:                                        
                                i +=1
                
            return combString, comb
        else:
            print('El conjunto de datos seleccionado no tiene ningún valor')
            
            return [], []
    
    def _init_dictData(self, input_data):
        
        if(len(input_data) > 0):
            
            dict_data = {}
            dict_dataDistr = {}
            
            for strg in self.arr_StringComb:
                dict_data[strg] = 0
                dict_dataDistr[strg] = []
                
            for data in input_data:
                key = ''
                
                for binary in data:
                    key += str(int(binary))                
                
                for strg in self.arr_StringComb:
                    if(key == strg):
                        dict_data[strg] += 1                                     
                        dict_dataDistr[key].append(data)                    
                
            return dict_data, dict_dataDistr
        
        else: 
            
            print('El conjunto de datos seleccionado no tiene ningún valor')
            
            return {}, {}
    """    
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
    """
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


