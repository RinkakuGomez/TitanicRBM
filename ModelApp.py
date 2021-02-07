# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 09:45:14 2020

@author: Pc
"""

import model.TitanicRBM
import os

class ModelApp:
    
    def __init__(self, dir_name, name_dataSet, num_epoch, n_visible, n_hidden, batch_size, learning_rate):
        
        self.arr_fString = ['Sex','Age','Survived', 'PClass']
        
        self.model = model.TitanicRBM.TitanicRBM(nameDataset = name_dataSet, n_epoch=num_epoch, lr=learning_rate, bs=batch_size, num_visible=n_visible, num_hidden=n_hidden, num_k=k, arr_fieldString=arr_fString)
        
        """
        *
        * Inicialización del modelo y preprocesado de los datos
        *
        """
        
        if(self.model.rbm.trained == False):
            
            print('Se ha iniciado el proceso de entrenamiento.\n\n')
            
            result = self._trainModel(name_filePrueba)
            
            if(result == 'OK'):
                print('El proceso de entrenamiento ha finalizado correctamente.\n\n')
                
        else:
            # Modelo ya entrenado
            print('El modelo ya ha sido entrenado.\n\n')
            
    def _trainModel(self, name_filePrueba):
        
        result = self.model.trainRBM(name_filePrueba)         
        
        return result
        
    def _infererModel(self, v):
        
        result = self.model.inferenceRBM(v)
        
        return result
    
    def trainProcess(self, name_filePrueba):
        
        if(app.model.rbm.trained == False):
           
            print('Iniciando proceso de entrenamiento.\n')
            
            result = app._trainModel(name_filePrueba)
            
            if(result == 'OK'):
                print('El proceso de entrenamiento ha finalizado correctamente.\n')
                print('Regresando al menú principal.\n\n')
                
                self.model.rbm.trained == True
                
        else:
            print('El modelo ya ha sido entrenado. ¿Desea volver a entrenar el modelo?\n')
            
            op = ''            
            
            while((op.lower() != 'si') and (op.lower() != 'no')):
                
                print('Introduzca Si, en caso de desear volver a entrenar el modelo, y No en caso contrario.\n')
                
                op = input('Introduzca la opción deseada:\n')                
                
                if(op.lower() == 'si'):
                    
                    print('Iniciando proceso de entrenamiento.\n\n')
        
                    result = app._trainModel(name_filePrueba)
                    
                    if(result == 'OK'):
                        print('El proceso de entrenamiento ha finalizado correctamente.\n')
                        print('Regresando al menú principal.\n\n')
                        
                        self.model.rbm.trained == True
                    
                elif(op.lower() == 'no'):
                    
                    print('Proceso de entrenamiento cancelado. Regresando al menú principal.\n\n')

                else:
                    print('La opción introducida no es correcta. Introduzca Si'
                          +'o NO en función de si desea volver a ejecutar el proceso de entrenamiento\n\n')
                    
    def inferenceProcess(self):
        
        if(app.model.rbm.trained == True):
            print('Iniciando proceso de inferencia.\n')
            
            op = 'no'  
            auxSex = auxAge = auxClass = auxSurvived = -1.0
            
            print('Introduzca el valor de los siguientes campos de entrada: \n')
            
            while(op.lower() != 'si'):                
                auxOp = ''

                for field in self.arr_fString:
                    val = False

                    # Valor campo Sex
                    if(field == 'Sex'):
                        
                        while(val == False):
                            sex = input('\t Introduzca el sexo de la persona (hombre o mujer):')
                            print('\n')
                            
                            if((sex.lower() == 'hombre') or (sex.lower() == 'mujer')):
                                val = True
                                
                                if(sex.lower() == 'hombre'):
                                    auxSex = 0
                                elif(sex.lower() == 'mujer'):
                                    auxSex = 1
                                else:
                                    auxSex = 0
                            else:
                                print('Error. Por favor, introduzca un valor correcto para el campo: '+field)
                    
                    # Valor campo Age        
                    elif(field == 'Age'):        
                        
                        while(val == False):
                            try:
                                age = int(input('\t Introduzca la edad de la persona (valor numérico entero):'))
                                print('\n')
                            
                            
                                val = True
                                if(float(age) > 18):
                                    auxAge = 1   
                                else:
                                    auxAge = 0
                            
                            except ValueError:
                                print('Error. Por favor, introduzca un valor correcto para el campo: '+field)
                    
                    # Valor campo PClass
                    elif(field == 'PClass'):        
                        
                        while(val == False):
                            ticketClass = input('\t Introduzca la clase de pasajero (1st, 2nd, 3rd o crew):')
                            print('\n')
                            
                            if((ticketClass.lower() == '1st') or (ticketClass.lower() == '2nd') or(ticketClass.lower() == '3rd') or (ticketClass.lower() == 'crew')):
                                val = True
                                
                                if(ticketClass == '1st'):
                                    auxClass = 1
                                elif(ticketClass == '2nd'):
                                    auxClass = 0
                                elif(ticketClass == 'crew'):
                                    auxClass = 0 #evaluar
                                elif(ticketClass == '3rd'):
                                    auxClass = 0 
                            else:
                                print('Error. Por favor, introduzca un valor correcto para el campo: '+field)

                    # Valor campo Survived
                    elif(field == 'Survived'):                    
                        auxSurvived = -1.0

                print('Array a inferir: '+str(auxSex)+', '+str(auxAge)+', '+str(auxClass)+', '+str(auxSurvived)+'\n')                
                    
                self._infererModel([auxSex, auxAge, auxClass, auxSurvived])
                
                
                while((auxOp.lower() != 'si') and (auxOp.lower() != 'no')):                    
                    
                    print('¿Desea finalizar de realizar inferencias sobre los '
                          +'datos y volver al menú principal (Si o No)?')
                    
                    auxOp = input('Introduzca la opción deseada:\n')
                    
                    if(op.lower() == 'no'):
                        op = auxOp
                        print('Introduzca los nuevos valores de los campos: \n')
                    
                    elif(op.lower() == 'si'):
                        op = auxOp
                        print('Regresando al menú principal.\n\n')
                    
                    else:
                        print('La opción introducida no es correcta. Introduzca Si' 
                          +'o NO, en función de si desea continuar el proceso de inferencia.\n\n')
        else:
            print('El modelo no ha sido entrenado previamente, por favor '
                  +'selecione la opción de entrenar el modelo en el menú para '
                  +'poder acceder a esta opción.\n\n')

if __name__== '__main__':
    
    #Cambiar valores por combinación óptima trás el análisis del modelo
    batch_size = 16
    learning_rate = 0.01
    n_hidden = 4
    
    
    n_visible = 4
    k = 1
    arr_fString = ['Sex','Age','Survived', 'PClass']
    num_epoch = 2000
    
    dir_name = 'resultadosTrain'
    name_dataSet = 'datasets/Titanic2200.csv'
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    name_filePrueba = dir_name+'/resultadoV'+str(n_visible)+'H'+str(n_hidden)+'E'+str(num_epoch)+'LR'+str(learning_rate)+'BS'+str(batch_size)+'.txt'
    app = ModelApp(dir_name, name_dataSet, num_epoch, n_visible, n_hidden, batch_size, learning_rate)  
    
    #Finaliza proceso train
    
    op = -1
    
    print('Bienvenido al menú de la aplicación\n\n')
    while op != 0:
    
        try:  
            
            print('Menú: \n')
            print('\t Opción 1- Realizar inferencia.\n')
            print('\t Opción 2- Volver a entrenar el modelo.\n')
            print('\t Opción 0- Salir de la aplicación.\n')
        
            op = int(input('Introduzca la opción deseada:\n'))
            
            if(int(op) == 1):
                app.inferenceProcess()
                os.system('clear')
        
            elif(int(op) == 2):
                app.trainProcess(name_filePrueba)
                os.system('clear')
            
            elif(int(op) == 0):
                print('Cerrando aplicación...\n\n')
                os.system('clear')
                
            else:
                os.system('clear')
                print('Opción incorrecta. Seleccione una opción de las especificadas en el menú.\n\n')
   
        except ValueError:
            os.system('clear')
            print('La opción introducida debe de ser númerica.\n\n')
        