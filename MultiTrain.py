# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:34:18 2020

@author: Pc
"""

import pruebasAnalisisEst


class LanzaderaTrain:
    
    def __init__(self):
        
        arr_BatchSize = [32, 16, 8]
        arr_LR = [0.1, 0.01, 1.0]
        arr_hidden = [8, 6] 
        num_epoch = 5000
        n_visible = 3
        k = 1
        arr_fString = ['Sex','Age','Survived']
        i = 1 
        for n_hidden in arr_hidden:
            for learning_rate in arr_LR:
                for batch_size in arr_BatchSize:
                    
                    print('Entrenamiento número: '+str(i))
                    name_CSVPrueba = 'pruebasPrecision/pruebaMultiH'+str(n_hidden)+'LR'+str(learning_rate)+'BS'+str(batch_size)+'.txt'
                    self.prueba = pruebasAnalisisEst.PruebasEstadistica(n_epoch=num_epoch, lr=learning_rate, bs=batch_size, num_visible=n_visible, num_hidden=n_hidden, num_k=k, arr_fieldString=arr_fString)
                    self.prueba.pruebaTest(name_CSVPrueba)
                    i+=1