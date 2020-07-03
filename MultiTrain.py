# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:34:18 2020

@author: Pc
"""

import pruebasAnalisisEst


    
if __name__== '__main__':
    
    arr_BatchSize = [64, 40, 32, 16, 8]
    arr_LR = [0.01, 1.0, 0.1, 0.001]
    arr_hidden = [2, 8, 4] 
    num_epoch = 10000
    n_visible = 4
    k = 1
    arr_fString = ['Sex','Age','Survived', 'PClass']
    i = 1 
    
    #prueba = pruebasAnalisisEst.PruebasEstadistica(n_epoch=1, lr=1.0, bs=32, num_visible=4, num_hidden=2, num_k=1, arr_fieldString=arr_fString)
    #prueba.prueba_ClassInput()
    for n_hidden in arr_hidden:
        for learning_rate in arr_LR:
            for batch_size in arr_BatchSize:
                
                print('Entrenamiento número: '+str(i))
                name_CSVPrueba = 'pruebasPrecision/pruebaMultiV'+str(n_visible)+'H'+str(n_hidden)+'E'+str(num_epoch)+'LR'+str(learning_rate)+'BS'+str(batch_size)+'.txt'
                prueba = pruebasAnalisisEst.PruebasEstadistica(n_epoch=num_epoch, lr=learning_rate, bs=batch_size, num_visible=n_visible, num_hidden=n_hidden, num_k=k, arr_fieldString=arr_fString)
                prueba.pruebaTest(name_CSVPrueba)
                i+=1
