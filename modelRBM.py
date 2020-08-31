# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Editor de Spyder

Este es un archivo temporal.
"""

import tensorflow as tf

#from flags import Flags

# Num Epoch
#tf.compat.v1.flags.DEFINE_integer('num_epoch', default = 1, help = 'Número de épocas', lower_bound = 1, flag_values = 50)

# Learning Rate
#tf.compat.v1.flags.DEFINE_float('learning_rate', default = 0.1, help = 'Ratio de aprendizaje', lower_bound = 0.001, upper_bound = 1.0, flag_values = 0.1)

# Batch Size
#tf.compat.v1.flags.DEFINE_integer('batch_size', default = 1, help = 'Tamaño del batch' , lower_bound = 1, flag_values = 80)

# Number visible
#tf.compat.v1.flags.DEFINE_integer('n_visible', default = 1, help = 'Número de nodos visibles', lower_bound = 1, flag_values = 2)

# Number hidden
#tf.compat.v1.flags.DEFINE_integer('n_hidden', default = 2, help = 'Número de nodos ocultos', lower_bound = 1, flag_values = 4)

# Número de ejecuciones del Gibb Sampling
#tf.compat.v1.flags.DEFINE_integer('k', default = 1, help = 'Variable de ejecución Gibb Sampling', lower_bound = 1, flag_values = 10)

#FLAGS = tf.compat.v1.flags

class RBM:
    
    """
    * Definición de la red neuronal: 
    *   visible_bias: contiene los valores de los bias de las neuronas de las entradas.
    *   hidden_bias: contiene los valores de los bias de las neuronas de la capa oculta.
    *   weights: contiene los pesos de las distintas entradas de cada neurona.
    """
    
    def __init__(self, num_epoch, learning_rate, batch_size, n_visible, n_hidden, k):
         
        """
        Definición de la sesión
        """
        self._sess = tf.compat.v1.Session()
        
        """
        Definición variable red neuronal
        """
    
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        
        # Variable de control
        self.trained = False
        
        """
        Definición de pesos y bias
        """    
        #self.visible_bias = tf.Variable(tf.random.uniform([1, self.n_visible]), name='visible_bias')
        self.visible_bias = tf.compat.v1.get_variable(name='visible_bias', shape=(1, self.n_visible), initializer=self._init_bias_zeros())
        #self.visible_bias = tf.compat.v1.get_variable(tf.compat.v1.zeros_like((1, self.n_visible), dtype=tf.float32), name='visible_bias')
        
        #self.hidden_bias = tf.Variable(tf.random.uniform([1, self.n_hidden]), name='hidden_bias')
        self.hidden_bias = tf.compat.v1.get_variable(name='hidden_bias', shape=(1, self.n_hidden), initializer=self._init_bias_zeros())
        #self.hidden_bias = tf.compat.v1.get_variable(tf.compat.v1.zeros_like((1, self.n_hidden), dtype=tf.float32), name='hidden_bias')
        
        """
        * Añadir std_dev como variable
        """
        #self.weights = tf.Variable(self._init_weight_distNormal(0.5), name='weights')
        self.weights = tf.compat.v1.get_variable(name='weights', shape=(self.n_visible, self.n_hidden), initializer=self._init_weight_distNormal(0.5))      
    
    """
    *
    * Métodos GS y Compute Gradients
    *
    """
      
    """
    *   Método: _hidden_sample 
    *   Descripción: Calcula una distribucción probabilistica a partir de los 
    *   datos introducidos en la variable visible_node
    *   visible_nodes: contiene los valores de las entradas que recibe la capa hidden
    """
    def _hidden_sample (self, visible_nodes):              
            
        multiply = tf.tensordot(visible_nodes, self.weights, axes=(-1,0))
        add = tf.add(multiply,self.hidden_bias)
        probability = tf.nn.sigmoid(add)
        
        h_bernouille = self._bernouille_sample(probability, shape=tf.shape(probability))
        
        return probability, h_bernouille
    
    """
    *   Método: _visible_sample 
    *   Descripción: Calcula una distribucción probabilistica a partir de los 
    *   datos introducidos en la variable hidden_node
    *   hidden_nodes: contiene los valores de las entradas que recibe la capa visible
    """
    
    def _visible_sample (self, hidden_nodes):              

        multiply = tf.tensordot(hidden_nodes, tf.transpose(self.weights,[1,0]), axes=(-1,0))
        add = tf.add(multiply,self.visible_bias)
        probability = tf.nn.sigmoid(add)
        
        v_bernouille = self._bernouille_sample(probability, shape=tf.shape(probability))
        
        return probability, v_bernouille
    
    def _bernouille_sample(self, probs, shape):
        
        return tf.where(
                tf.less(probs,tf.random.uniform(shape,minval=0.0, maxval=1.0)),
                x=tf.zeros_like(probs),
                y=tf.ones_like(probs))
    
        
    """
    * Método: _gibbsSampling_v
    * Descripción:
    * Variables: 
    *       - v: visible input vector
    """
    def _gibbsSampling_v(self, v):
        
    
        def condition(i, Vk, Hk, v):
            
            result = tf.less(i, self.k)            
                
            return result
    
        def body(i, Vk, Hk, v):
            
            # Sampling hidden and visible
            _,vector_h = self._hidden_sample(v)
            _,vector_v = self._visible_sample(vector_h)
            
            
            return [i+1, vector_v, vector_h, v]        
        
        ph0,_ = self._hidden_sample(v)
        
        i = 0
        
        print(self._sess.run(tf.less(i, self.k)))
        Vk = v
        Hk = tf.zeros_like(ph0)
        
        [i, Vk, Hk, v] = tf.while_loop_v2(condition, body, [i, Vk, Hk, v])

        
        phk = self._hidden_sample(Vk)
        
        return v, Vk, ph0, phk 
    
    """
    * Método: _gibbsSampling_v2
    * Descripción:
    * Variables: 
    *       - v: visible input vector
    """
    def _gibbsSampling_v2(self, v):
        
        i = 0
        gibb_sampling = []
        
        while i <= self.k:
            
            ph0, h0 = self._hidden_sample(v)
            phI, vI = self._visible_sample(h0)
                        
            if(i == 0 or i == self.k):

                gibb_sampling.append([v, ph0])
            
            v = vI
            i = i + 1
                
        return gibb_sampling 
    
    """
    * Método: _updateParams
    * Descripción: asigna el nuevo valor a los pesos y bias de la red en función
    * de las diferencias calculadas.
    * Variables: 
    *       - v: visible input vector
    """    
    def updateParams(self, dW, dbh, dbv):
        
        #Aplicar el alpha sobre la diferencia de los gradientes
        
        XW = tf.divide(tf.math.reduce_sum(dW), self.batch_size)
        Xbh = tf.divide(tf.math.reduce_sum(dbh), self.batch_size)
        Xbv = tf.divide(tf.math.reduce_sum(dbv), self.batch_size)
        
        # Assign deprecate
        update_param = [tf.compat.v1.assign(self.weights, tf.add(self.weights, self.learning_rate*XW)),
                        tf.compat.v1.assign(self.hidden_bias, tf.add(self.hidden_bias, self.learning_rate*Xbh)),
                        tf.compat.v1.assign(self.visible_bias, tf.add(self.visible_bias, self.learning_rate*Xbv))]
                
        return update_param
    
    
    """
    * Método: _computeGradientes
    * Descripción:
    * Variables: 
    *       - v: visible input vector
    """    
    """def _computeGradientes(self, V0, ph0, Vk, phk):
        
        def condition(i, V0, ph0, Vk, phk, dW, dbh, dbv):
            
            result = tf.less(i, self.k)
            
            return result[0]
        
        def body(i, V0, ph0, Vk, phk, dW, dbh, dbv):
            
            v0_gradient = V0[i]
            ph0_gradient = ph0[i]
            
            Vk_gradient = Vk[i]
            phk_gradient = phk[i]
                        
            #Shape para multiplicar
            v0_gradient = tf.reshape(v0_gradient, [self.n_visible, 1])
            ph0_gradient = tf.reshape(ph0_gradient, [1, self.n_hidden])
            
            Vk_gradient = tf.reshape(Vk_gradient, [self.n_visible, 1])
            phk_gradient = tf.reshape(phk_gradient, [1, self.n_hidden])
            
            #Cal gradientes
            dW__gradient = tf.subtract(tf.matmul(v0_gradient, ph0_gradient),tf.matmul(Vk_gradient, phk_gradient))
            dbh_gradient = tf.subtract(ph0_gradient, phk_gradient)
            dbv_gradient = tf.subtract(v0_gradient, Vk_gradient)
            
            
            return dW__gradient, dbh_gradient, dbv_gradient 
                
        i = 0
        
        dW = tf.compat.v1.zeros_like((self.n_visible, self.n_hidden))
        dbh = tf.compat.v1.zeros_like((self.n_hidden))
        dbv = tf.compat.v1.zeros_like((self.n_visible))
        
        [i, V0, ph0, Vk, phk, dW, dbh, dbv] = tf.while_loop(condition, body, [i, V0, ph0, Vk, phk, dW, dbh, dbv])        
        
        return dW, dbh, dbv
    """
    
    """
    * Método: _computeGradientes
    * Descripción:
    * Variables: 
    *       - v: visible input vector
    """    
    def _computeGradientes_v2(self, V0, ph0, Vk, phk):        
                                                
        dW = tf.compat.v1.zeros_like((self.n_visible, self.n_hidden))
        dbh = tf.compat.v1.zeros_like((self.n_hidden))
        dbv = tf.compat.v1.zeros_like((self.n_visible))
                    
        #Shape para multiplicar
        V0 = tf.reshape(V0, [self.n_visible, 1])
        ph0 = tf.reshape(ph0, [1, self.n_hidden])
        
        Vk = tf.reshape(Vk, [self.n_visible, 1])
        phk = tf.reshape(phk, [1, self.n_hidden])
        
        #Cal gradientes
        dW = tf.subtract(tf.matmul(V0, ph0),tf.matmul(Vk, phk))
        dbh = tf.subtract(ph0, phk)
        dbv = tf.subtract(V0, Vk)
                    
        return dW, dbh, dbv
    
    """
    * Método: _training
    * Descripción:
    * Variables: 
    *       - v: visible input vector
    """    
    def training(self, v):        
        
        gibb_S = self._gibbsSampling_v2(v)
        
        i = 0      

        for x in gibb_S:
            
            if(i == 0):
                                
                v0 = x[0]
                
                ph0 = x[1]
                
            elif(i == 1):
                
                vk = x[0]
                
                phk = x[1]
                
            i+=1
        
        dW, dbh, dbv = self._computeGradientes_v2(v0, ph0, vk, phk)        
        
        return dW, dbh, dbv, v0, ph0, vk, phk
    
                           
    """
    * Método: _inference
    * Descripción:
    * Variables: 
    *       - v: visible input vector
    """    
    def inference(self, v):
        
        phv,h_ = self._hidden_sample(v)
        
        pvh,v_ = self._visible_sample(h_)
        
        return phv,h_,pvh,v_
    
    """
    *
    * Métodos de init Bias y Weight
    *
    """

    """
    * Método: _init_weigth_distNormal
    * Descripción:
    * Variables: 
    *       - v: visible input vector
    """
    
    def _init_weight_distNormal(self, std_dev):
                
        return tf.random_normal_initializer(mean=0.0, stddev=std_dev)
    
    
    """
    * Método: _init_weigth_dist
    * Descripción:
    * Variables: 
    *       - v: visible input vector
    """
    
    def _init_weight_distUniforme(self):
        
        return tf.random.uniform([self.n_visible, self.n_hidden])
    
    
    """
    * Método: _init_bias_zeros
    * Descripción:
    * Variables: 
    *       - v: visible input vector
    """
    
    def _init_bias_zeros(self):
        
        return tf.zeros_initializer()