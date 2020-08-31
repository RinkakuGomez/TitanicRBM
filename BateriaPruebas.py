# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:27:55 2020

@author: Pc
"""

from modelRBM import RBM
import tensorflow as tf
import numpy as np
import csv

class BateriaPruebas:
    def __init__(self, nameDataset, n_epoch, lr, bs, num_visible, num_hidden, num_k, arr_fieldString):
        
        #Titanic dataset 1300
        self.nameCSV = nameDataset
        input_data = self._readCSVdata(self.nameCSV, arr_fieldString)
                      
        if(len(input_data) > 0):

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
            
            
            # Contiene la distribución de los datos y los datos separados 
            # por tipo de valores para los data set de train y test
            self.dict_trainComb, self.dict_trainDist = self._init_dictData(arr_train)
            self.dict_testComb, self.dict_testDist = self._init_dictData(arr_test)        
            
            # Dividimos los data set de train y test en batch        
            self.trainingData = self._create_batch(np.random.permutation(arr_train))
            self.testData = self._create_batch(np.random.permutation(arr_test))
        
        else:
            print('El fichero del data set seleccionado no contiene ningún registro')
    
    """
    Prueba readCSV división data set train y test, y división batch 
    """
    def prueba_readCSV(self):        
        
        if((len(self.trainingData) > 0) or (len(self.testData) > 0)):
            i = 1   
            print('Data set de entrenamiento\n')
            for batch_data in self.trainingData:            
                print('Batch número: '+str(i)+'\n')
                
                for data in batch_data:
                    print('Valor dato de entrada: '+str(data))
                i+=1
                
            i = 1   
            print('Data set de test\n')
            for batch_data in self.testData:            
                print('Batch número: '+str(i)+'\n')
                
                for data in batch_data:
                    print('Valor dato de entrada: '+str(data))
                i+=1
            
            return 'OK'
        
        else:
            print('Los data set de entrenamiento o test están vacíos\n\n')
            
            return 'fail'
    
    """
    Prueba combinatoria datos de entrada e inicialización de diccionarios
    """
    def prueba_CombDict(self):        
        
        if((len(self.arr_StringComb) > 0) or (len(self.arr_Comb) > 0)):
            
            print('\nArray string valores combinatoria')
            for str_comb in self.arr_StringComb:
                print('Valor: '+str(str_comb))
                        
            print('Array valores combinatoria')
            for data_comb in self.arr_Comb:        
                print('Valor: '+str(data_comb))
            
            print('\n')
            
        else:
            print('Error al inicializar los arrays con las combinatorias\n\n')
            
        input('Pulse [intro] para continuar:')
        
        if((len(self.dict_dataComb) > 0) or (len(self.dict_dataDist) > 0)):
            
            print('Distribución data set completo')
            for key, prob in self.dict_dataComb.items():
                print('Valor key: '+str(key)+' distribución: '+str(prob))
                        
            print('\nData set completo')
            for key, prob in self.dict_dataDist.items():
                print('Valor key: '+str(key)+' data set: '+str(prob))
                
            print('\n')
        else:
            print('Error al inicializar los diccionarios del data set completo.')
        
        input('Pulse [intro] para continuar:')
        
        if((len(self.dict_trainComb) > 0) or (len(self.dict_trainDist) > 0)):
            
            print('Distribución train data set')
            for key, prob in self.dict_trainComb.items():
                print('Valor key: '+str(key)+' distribución: '+str(prob))
                        
            print('\nTrain data set')
            for key, prob in self.dict_trainDist.items():
                print('Valor key: '+str(key)+' data set: '+str(prob))
            
            print('\n')
            
        else:
            print('Error al inicializar los diccionarios del train data set.')
        
        input('Pulse [intro] para continuar:')
        
        if((len(self.dict_testComb) > 0) or (len(self.dict_testDist) > 0)):
            
            print('Distribución test data set')
            for key, prob in self.dict_testComb.items():
                print('Valor key: '+str(key)+' distribución: '+str(prob))
                        
            print('\nTest data set')
            for key, prob in self.dict_testDist.items():
                print('Valor key: '+str(key)+' data set: '+str(prob))
            
            print('\n')
        
        else:
            print('Error al inicializar los diccionarios del test data set.')
        
        return 'OK'
        
    """
    Prueba Hidden y Visible sample única iteracción
    """
    def pruebaGS_Hidden(self):
        
        prob, bernouille = self.prueba._hidden_sample([1.0,0.0])

        print('Prueba prob: '.format(prob))
        print('Prueba bernouille: '.format(bernouille))
        print('Prueba shape: '.format(tf.shape(prob)))
        
        prob, bernouille = self.prueba._visible_sample(bernouille)

        print('Prueba prob2: '.format(prob))
     
        print('Prueba bernouille2: '.format(bernouille))
        
    """
    Prueba Gibb Sampling y Compute Gradient única iteracción
    """
    def pruebaGS(self):
        
        v0, ph0, vk, phk, dW, dbh, dbv = self.prueba.training([1.0,0.0,0.0,-1.0])
                        
        print('Prueba GS: ')
        
        print('V0: ')
        print(v0)

        print('PH0: {}'.format(ph0))
        
        print('Prueba GS 2: ')
        
        print('VK: {}'.format(vk))

        print('PHK: {}'.format(phk))
        
        print('Prueba diff: ')

        print('dW: {}'.format(dW))
   
        print('dbh: {}'.format(dbh))
        
        print('dbv: {}'.format(dbv))

    """
    * Método: pruebaInferencia
    * Descripción: Prueba proceso inferencia.
    * Variables: 
    *       - v: array con los datos a inferir.
    """
    def pruebaInferencia(self, v):
        
        phv,h_,pvh,v_ = self.prueba.inference(v)
        
        print('Valor inicial v: '+str(v))
        print('Valor phv: '+str(phv))
        
        print('Valor v bernouille: '+str(v_))
        print('Valor pvh: '+str(pvh))
        
    """
    * Método: pruebaTest
    * Descripción: Prueba procesos train y test
    * Variables: 
    *       - name_FilePrueba: string con el nombre del fichero generado con los resultados.
    """ 
    def pruebaTest(self, name_FilePrueba):
        
        
        if((len(self.trainingData) == 0) or (len(self.testData) == 0)):
            
            print('El conjunto de datos de entrenamiento está vacío. El tamaño es: '+ str(len(self.trainingData)))
            print('El conjunto de datos de test está vacío. El tamaño es: '+ str(len(self.testData)))
                
        else:
            with open(name_FilePrueba, 'w+') as file:  
                file.write('Prueba Multi Entrenamiento\n')
                
                file.write('\tValores de la red:\n')
                file.write('\t\tNum epoch: '+str(self.prueba.num_epoch)+'\t\t LR: '+str(self.prueba.learning_rate)+'\n')
                file.write('\t\tNum visible: '+str(self.prueba.n_visible)+'\t\t Num hidden: '+str(self.prueba.n_hidden)+'\n')
                file.write('\t\tBatch size: '+str(self.prueba.batch_size)+'\t\t K: '+str(self.prueba.k)+'\n')
                
                pesos_init = self.prueba.weights
                biasV_init = self.prueba.visible_bias
                biasH_init = self.prueba.hidden_bias            
                                
                probs_dictTest = {}
                probs_dictTrain = {}
                
                for epoch in range(1, self.prueba.num_epoch+1):            
                                        
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
                            bool_mask = tf.cast(mask, dtype=tf.bool)
                            
                            #Calculo de accuracy
                            acc = tf.where(bool_mask, x=tf.abs(tf.subtract(v0,vk)), y=tf.zeros_like(v0))
                            n_values = tf.math.reduce_sum(mask)
                            
                            accuracy_dataTrain += tf.subtract(1.0, tf.divide(tf.math.reduce_sum(acc), n_values))
                        
                            i += 1
                        
                        accuracy_batchTrain += accuracy_dataTrain/i   
                        j += 1
                                            
                        #Actualizamos los parámetros
                        self.prueba.updateParams(arr_dW, arr_dBh, arr_dBv)

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
                        bool_mask = tf.cast(mask, dtype=tf.bool)
                        
                        #Calculo de accuracy
                        acc = tf.where(bool_mask, x=tf.abs(tf.subtract(data_n,v_)), y=tf.zeros_like(data_n))
                        n_values = tf.math.reduce_sum(mask)
                        
                        accuracy_dataTest += tf.subtract(1.0, tf.divide(tf.math.reduce_sum(acc), n_values))
                        
                        i += 1
                    
                    accuracy_batchTest += accuracy_dataTest/i        
                    j += 1
                    
                    accuracy_test = accuracy_batchTest/j
                    
                    print('Precisión test: {}'.format(accuracy_test))
                    file.write('\t\tPrecisión test: {}\n'.format(accuracy_test))
                print('Fin proceso test')
                    
                for batch_n in self.trainingData:
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
        return ''
    
    """            
    Métodos AUX
    """
    
    """
    * Método: _readCSVdata
    * Descripción: Método lectura fichero CSV.
    * Variables: 
    *       - namecsv: string con el nombre del fichero CSV con el data set.
    *       - arr_fieldString: array con los nombres de las variables a usar.
    """
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
      
    """
    * Método: _getCombinatorial
    * Descripción: Método obtención combinación de valores del data set.
    * Variables: 
    *       - None
    """    
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
    
    """
    * Método: _init_dictData
    * Descripción: Método inicialización diccionarios.
    * Variables: 
    *       - input_data: lista con el conjunto de datos.
    """
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
    * Método: _create_batch
    * Descripción: Método división en batch data set.
    * Variables: 
    *       - dataArr: lista con el conjunto de datos a dividir en batch.
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

    """
    * Método: _classInput
    * Descripción: Método empleado para determinar el valor binario que 
    * corresponde a cada uno de los tipos de la clase existentes en función de 
    * la proporción de supervivientes/ahogamientos que este presenta.
    * Variables: 
    *       - None
    """        
    def _classInput(self):
        
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