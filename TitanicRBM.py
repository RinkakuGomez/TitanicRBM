# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import tensorflow as tf
import csv
import numpy as np

class RBM_TitacnicSex:
    
    def __init__(self, n_visible, n_hidden, n_iteraciones, learning_rate, name_file):
        
        """
        Creamos el grafo
        """
        self._graph = tf.Graph()
        
        """
        Inicializamos el grafo
        """
        
        with self._graph.as_default():
        
            self.n_iteraciones = n_iteraciones
            
            """
            Definición de pesos y bias
            """
            
            self.visible_bias = tf.Variable(tf.random_uniform([1,n_visible]), name='visible_bias')
            self.hidden_bias = tf.Variable(tf.random_uniform([1,n_hidden]), name='hidden_bias')
            
            self.weights = tf.Variable(tf.random_normal((n_visible,n_hidden), mean=0.0, stddev=0.01), name='weights')
            
            """
            Definición placeholder input y ratio de aprendizaje
            """
            
            self._input = tf.placeholder(tf.float32, [n_visible], name='input')
            self._lerning_rate = tf.Variable(tf.fill([n_visible, n_hidden], learning_rate), name='learning_rate')
            
            """
            Definición de las variables state
            """
            
            self._hidden_state = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name='hidden_state'))
            
            """
            Definición de variables Gibs Sampling
            """
            
            input_data = self.readCSV(name_file)
            input_matrix = tf.transpose(tf.stack([self._input for i in input_data]))
            
            _prob_hidden = tf.sigmoid(tf.add(tf.multiply(self.weights, input_matrix), tf.stack(self.hidden_bias[0] for i in range(n_visible))))
            self._hidden_state = self.calc_state(_prob_hidden)
            
            """
            _prob_visible = tf.sigmoid(tf.add(tf.multiply(self._hidden_state, self.weights), tf.transpose(tf.stack([self.visible_bias[0] for i in range(n_hidden)]))))
            self.visible_state = self.calc_state(_prob_visible)
            """
            
            """
            Inizalización de variables y sesión
            """
            self._sess = tf.Session()
            initialization = tf.global_variables_initializer()
            self._sess.run(initialization)
            
    def calc_state(self, probability):
        return tf.floor(probability + tf.random_uniform(tf.shape(probability), 0, 1))
    
    def train_RBM(self, lerning_rate):
        tf.train.GradientDescentOptimizer(self._learning_rate)
        
    def readCSV(self, namecsv):
        with open(namecsv, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            i = 0
            fields = []
            data = []
            for row in csvreader:
                
                if(i==0):
                    fields.append(row[0])
                    fields.append(row[1])

                else:
                    data.append([row[0],row[1]])
                
                i = i + 1
            return data