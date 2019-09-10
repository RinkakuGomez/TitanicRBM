# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:25:38 2019

@author: Pc
"""

import csv

class pruebaCSV():
    
    def readCSV(self, namecsv):
        with open(namecsv, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            i = 0
            fields = []
            valores = []
                        
            for row in csvreader:
                
                if(i==0):
                    fields.append(row[0])
                    fields.append(row[1])

                else:
                    
                    valores.append([row[0],row[1]])
                
                i = i + 1  
                        
            return valores
