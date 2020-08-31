# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:34:18 2020

@author: Pc
"""

import TitanicRBM
import os
    
if __name__== '__main__':
    
    arr_BatchSize = [64, 40, 32, 16, 8]
    arr_LR = [0.01, 1.0, 0.1, 0.001]
    arr_hidden = [2, 8, 4] 
    num_epoch = 2000
    n_visible = 4
    k = 1
    arr_fString = ['Sex','Age','Survived', 'PClass']
    i = 1 
    name_dataSet = 'Titanic2200.csv'
    dir_name = 'pruebasPrecision2200'
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
        
    for n_hidden in arr_hidden:
        for learning_rate in arr_LR:
            for batch_size in arr_BatchSize:
                
                print('Entrenamiento número: '+str(i))
                name_filePrueba = dir_name+'/pruebaMultiV'+str(n_visible)+'H'+str(n_hidden)+'E'+str(num_epoch)+'LR'+str(learning_rate)+'BS'+str(batch_size)+'.txt'
                prueba = TitanicRBM.TitanicRBM(nameDataset = name_dataSet,n_epoch=num_epoch, lr=learning_rate, bs=batch_size, num_visible=n_visible, num_hidden=n_hidden, num_k=k, arr_fieldString=arr_fString)
                prueba.trainRBM(name_filePrueba)
                i+=1
