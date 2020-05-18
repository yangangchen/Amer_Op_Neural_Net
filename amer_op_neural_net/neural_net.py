'''
The Neural Network object for the American Option pricing and hedging algorithm
'''

from .config import *
from .options import levelset_function, payoff_function


class NeuralNet:
    def __init__(self, NN_name, dim_layers):
        """
        Constructor of the NeuralNet class
	
        NN_name: string, name of the neural network graph
        dim_layers: list of integers, dimensions of the hidden layers
        """

        self.NN_name = NN_name
        self.additional_features = ("P", "Y")
        self.dim_layers = dim_layers
        self.dim_layers[0] = len(self.additional_features)
        self.dim_layers[-1] = 1
        self.num_layers = len(self.dim_layers)

        with tf.variable_scope(self.NN_name, reuse=tf.AUTO_REUSE):
            self.build_hyperparams()
            self.build_dataset()

            ## Definition and initialization of the trainable variables in the network
            self.build_graph_variables()
            ## Construction of the training graph
            self.Y, self.gradY = \
            	self.build_graph(graph_type="train", X=self.train_X, Y_prestep=self.train_Y_prestep,
                                 gradY_prestep=self.train_gradY_prestep, sharpness=sharpness)
            ## Definition of the loss function and the optimizer
            self.loss(dA=self.dA, YLabel=self.YLabel, Y=self.Y, gradY=self.gradY)
            self.optimizer()

            ## Construction of the test graph, which shares the same trainable variable of the training graph
            self.ensemble_Y, self.ensemble_gradY, self.input_mv_init_op = \
            	self.build_graph(graph_type="test", X=self.test_X, Y_prestep=self.test_Y_prestep,
                                 gradY_prestep=self.test_gradY_prestep, sharpness=sharpness)

        self.check_graph()

    def build_hyperparams(self):
        """
        Definition of training hyperparameters
        """
        self.step = tf.get_variable("step", dtype=tf.int32, shape=[],
                                    initializer=tf.constant_initializer(0), trainable=False)
        self.init_rate = tf.get_variable("init_rate", dtype=tf_floattype, shape=[],
                                         initializer=tf.constant_initializer(0.0), trainable=False)
        self.decay_rate = tf.get_variable("decay_rate", dtype=tf_floattype, shape=[],
                                          initializer=tf.constant_initializer(1.0), trainable=False)
        self.n_relaxstep = tf.get_variable("n_relaxstep", dtype=tf.int32, shape=[],
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.n_decaystep = tf.get_variable("n_decaystep", dtype=tf.int32, shape=[],
                                           initializer=tf.constant_initializer(0), trainable=False)
        ## self.rate: Global learning rate of the Adam optimizer
        self.rate = tf.cast(tf.train.exponential_decay(
            self.init_rate, global_step=tf.clip_by_value(self.step - self.n_relaxstep, 0, self.n_decaystep),
            decay_steps=self.n_decaystep, decay_rate=self.decay_rate, staircase=False), dtype=tf_floattype)
        ## self.bn_rate: Learning rate of the batch normalization
        bn_init_rate = 1.0
        self.bn_rate = bn_init_rate * (tf.cast(tf.train.exponential_decay(
            bn_init_rate, global_step=tf.clip_by_value(self.step, 0, self.n_decaystep),
            decay_steps=self.n_decaystep, decay_rate=self.decay_rate, staircase=False), dtype=tf_floattype)
                                       - self.decay_rate) / (1 - self.decay_rate)

    def build_dataset(self):
        '''
        Define input tensor placeholders, training input tensors and test input tensors
        '''
        self.ph_X = tf.placeholder(dtype=tf_floattype, shape=[None, None, d], name="X")
        self.ph_dA = tf.placeholder(dtype=tf_floattype, shape=[None, None, d], name="dA")
        self.ph_YLabel = tf.placeholder(dtype=tf_floattype, shape=[None, None, 1], name="YLabel")
        self.ph_Y_prestep = tf.placeholder(dtype=tf_floattype, shape=[None, None, 1], name="Y_prestep")
        self.ph_gradY_prestep = tf.placeholder(dtype=tf_floattype, shape=[None, None, d, 1], name="gradY_prestep")

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.ph_X, self.ph_dA, self.ph_YLabel, self.ph_Y_prestep, self.ph_gradY_prestep)).batch(batch_size)
        train_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self.train_init_op = train_iter.make_initializer(train_dataset)
        self.train_X, self.dA, self.YLabel, self.train_Y_prestep, self.train_gradY_prestep = train_iter.get_next()

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.ph_X, self.ph_Y_prestep, self.ph_gradY_prestep)).batch(tf.cast(tf.shape(self.ph_X)[0], tf.int64))
        test_iter = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
        self.test_init_op = test_iter.make_initializer(test_dataset)
        self.test_X, self.test_Y_prestep, self.test_gradY_prestep = test_iter.get_next()

    def __batch_norm_layer_variables(self, batch_norm_name, dim):
        '''
        Define and initialize the trainable variables of a batch normalization layer, which will become part of the neural network trainable variables

        batch_norm_name: string, the name of the batch normalization layer
        dim: int, dimension of the batch normalization layer (= dimension of the hidden layer)
        '''
        with tf.variable_scope(batch_norm_name, reuse=tf.AUTO_REUSE):
            tf.get_variable("mv_mean", dtype=tf_floattype, shape=[1, num_channels, dim],
                            initializer=tf.constant_initializer(0.0), trainable=False)
            tf.get_variable("mv_var", dtype=tf_floattype, shape=[1, num_channels, dim],
                            initializer=tf.constant_initializer(1.0), trainable=False)
            tf.get_variable("beta", dtype=tf_floattype, shape=[1, num_channels, dim],
                            initializer=tf.constant_initializer(0.0), trainable=True)
            tf.get_variable("gamma", dtype=tf_floattype, shape=[1, num_channels, dim],
                            initializer=tf.constant_initializer(1.0), trainable=True)

    def __batch_norm_layer(self, batch_norm_name, Z, input_mv=False, tol=1e-8):
        '''
        Define a batch normalization layer, which will become part of the neural network

        batch_norm_name: string, the name of the batch normalization layer
        Z: hidden layer value
        input_mv: boolean, whether a batch normalization layer is an input normalization (or a hidden layer normalization)
        '''
        with tf.variable_scope(batch_norm_name, reuse=tf.AUTO_REUSE):
            mv_mean = tf.get_variable("mv_mean", dtype=tf_floattype)
            mv_var = tf.get_variable("mv_var", dtype=tf_floattype)
            beta = tf.get_variable("beta", dtype=tf_floattype)
            gamma = tf.get_variable("gamma", dtype=tf_floattype)
            batch_mean, batch_var = tf.nn.moments(Z, axes=list(range(len(Z.get_shape())-2)), keep_dims=True)
            if input_mv:
                input_mv_init_op = [tf.assign(mv_mean, batch_mean), tf.assign(mv_var, batch_var)]
                Z = tf.nn.batch_normalization(Z, mv_mean, mv_var, beta, gamma, tol)
                return input_mv_init_op, Z
            else:
                train_mv_mean = tf.assign(mv_mean,
                                          mv_mean * (1.0 - self.bn_rate) + batch_mean * self.bn_rate)
                train_mv_var = tf.assign(mv_var,
                                         mv_var * (1.0 - self.bn_rate) + batch_var * self.bn_rate)
                with tf.control_dependencies([train_mv_mean, train_mv_var]):
                    Z = tf.nn.batch_normalization(Z, mv_mean, mv_var, beta, gamma, tol)
                return Z

    def __tensor_chainrule(self, gradZ1, gradZ2):
        return tf.expand_dims(gradZ1, axis=-2) * gradZ2

    def __tensor_contract(self, X, W):
        dim0 = tf.shape(X)[0]
        Wtile = tf.tile(tf.expand_dims(W, axis=0), (dim0, 1, 1, 1))
        return tf.squeeze(tf.matmul(tf.expand_dims(X, axis=-2), Wtile), axis=-2)

    def build_graph_variables(self, stddev=1.0):
        """
	Define and initialize all the trainable variables of each network
        """
        #### Input layers
        layer = 0
        with tf.variable_scope("layer" + str(layer), reuse=tf.AUTO_REUSE):
            self.__batch_norm_layer_variables("batchX", d)
            self.__batch_norm_layer_variables("batchZ", self.dim_layers[0])

        #### Hidden layers
        for layer in range(1, self.num_layers - 1):
            with tf.variable_scope("layer" + str(layer), reuse=tf.AUTO_REUSE):
                if layer == 1:
                    lim = stddev / math.sqrt(d + self.dim_layers[1])
                    WX = tf.get_variable("WX", dtype=tf_floattype,
                                         shape=[num_channels, d, self.dim_layers[1]],
                                         initializer=tf.random_uniform_initializer(-lim, lim), trainable=True)
                    lim = stddev / math.sqrt(self.dim_layers[0] + self.dim_layers[1])
                    WZ = tf.get_variable("WZ", dtype=tf_floattype,
                                         shape=[num_channels, self.dim_layers[0], self.dim_layers[1]],
                                         initializer=tf.random_uniform_initializer(-lim, lim), trainable=True)

                if layer >= 2:
                    lim = stddev / math.sqrt(self.dim_layers[layer - 1] + self.dim_layers[layer])
                    WZ = tf.get_variable("WZ", dtype=tf_floattype,
                                         shape=[num_channels, self.dim_layers[layer - 1], self.dim_layers[layer]],
                                         initializer=tf.random_uniform_initializer(-lim, lim), trainable=True)

                self.__batch_norm_layer_variables("batchZ", self.dim_layers[layer])

        #### Output layer
        self.delta = tf.get_variable("delta", dtype=tf_floattype, shape=[],
                                     initializer=tf.constant_initializer(1.0), trainable=False)
        self.alpha = tf.get_variable("alpha", dtype=tf_floattype, shape=[1, num_channels, 1],
                                     initializer=tf.constant_initializer(1.0), trainable=True)
        self.ensemble_alpha = tf.reduce_mean(self.alpha)

        layer = self.num_layers - 1
        with tf.variable_scope("layer" + str(layer), reuse=tf.AUTO_REUSE):
            lim = stddev / math.sqrt(self.dim_layers[layer - 1] + 1)
            self.WZ = tf.get_variable("WZ", dtype=tf_floattype,
                                      shape=[num_channels, self.dim_layers[layer - 1], 1],
                                      initializer=tf.random_uniform_initializer(-lim, lim), trainable=True)
            self.b = tf.get_variable("b", dtype=tf_floattype, shape=[1, num_channels, 1],
                                     initializer=tf.constant_initializer(0.0), trainable=True)
            self.ensemble_b = tf.reduce_mean(self.b)

    def build_graph(self, graph_type, X, Y_prestep, gradY_prestep, sharpness):
        """
        Construct the network
        """
        assert graph_type == "train" or graph_type == "test"

        #### Input layers
        layer = 0
        with tf.variable_scope("layer" + str(layer), reuse=tf.AUTO_REUSE):
            L = levelset_function(X, sharpness=sharpness)

            input_mv_init_op, X_normal = self.__batch_norm_layer("batchX", X, input_mv=True)

            Z = tf.concat([Y_prestep, L], axis=-1)
            input_mv_init_op_Z, Z_normal = self.__batch_norm_layer("batchZ", Z, input_mv=True)
            input_mv_init_op += input_mv_init_op_Z

            input_mv_init_op = tf.group(input_mv_init_op)

        #### Hidden layers
        for layer in range(1, self.num_layers - 1):
            with tf.variable_scope("layer" + str(layer), reuse=tf.AUTO_REUSE):
                ## Linear transformation
                if layer == 1:
                    WX = tf.get_variable("WX", dtype=tf_floattype)
                    WZ = tf.get_variable("WZ", dtype=tf_floattype)
                    Z = self.__tensor_contract(X_normal, WX) + self.__tensor_contract(Z_normal, WZ)

                if layer >= 2:
                    WZ = tf.get_variable("WZ", dtype=tf_floattype)
                    Z = self.__tensor_contract(Z, WZ)

                ## Batch normalization
                Z = self.__batch_norm_layer("batchZ", Z)

                ## Nonlinear activation
                Z = tf.nn.relu(Z)

        #### Output layer
        layer = self.num_layers - 1

        Z = self.__tensor_contract(Z, self.WZ) + self.b
        Y = self.alpha * (Y_prestep + self.delta * dt * Z)

        gradY = tf.expand_dims(tf.gradients(Y, X)[0], axis=-1) \
                + self.__tensor_chainrule(tf.gradients(Y, Y_prestep)[0], gradY_prestep)
        
        if graph_type == "train":
            return Y, gradY
        elif graph_type == "test":
            ensemble_Y = tf.reduce_mean(Y, axis=1, keepdims=True)
            ensemble_gradY = 1 / num_channels * tf.reduce_mean(gradY, axis=1, keepdims=True)
            return ensemble_Y, ensemble_gradY, input_mv_init_op

    def loss(self, dA, YLabel, Y, gradY):
        self.trainable_variables = tf.trainable_variables(scope=self.NN_name)
        dA = tf.expand_dims(dA, axis=-1)
        Res = YLabel - Y - math.exp(-r * dt) * tf.reduce_sum(dA * gradY, axis=-2)
        self.Loss = tf.reduce_mean(Res ** 2)

    def optimizer(self):
        self.Optimizer = tf.train.AdamOptimizer(learning_rate=self.rate)
        # self.Optimizer = tf.train.RMSPropOptimizer(learning_rate=self.rate)
        Grad_Var_list = self.Optimizer.compute_gradients(self.Loss, var_list=self.trainable_variables)
        Grad_Var_list = [(tf.clip_by_value(Grad, -10, 10), Var)
                         if "/alpha:0" not in Var.name else (Grad, Var)
                         for Grad, Var in Grad_Var_list]
        self.Optimizer = self.Optimizer.apply_gradients(Grad_Var_list, global_step=self.step)

    def check_graph(self):
        '''
        Print out the graph for examination
        '''
        print("\n Trainable variables in the main neural network ", self.NN_name, ": ")
        for Var in self.trainable_variables:
            print("   ", Var)
        print("\n Global variables in the main neural network ", self.NN_name, ": ")
        for Var in tf.global_variables():
            print("   ", Var)

    def fit_Optimizer(self, sess, Xn, dAn, YLabeln, Y_prestepn, gradY_prestepn, n_totalstep, n_unitstep):
        '''
        Train a neural network using the previously defined optimizer
        '''
        assert self.n < N
        print("\n **** ", self.NN_name, ", Adam Optimization : ")
        #### Data input 
        sess.run(self.train_init_op, feed_dict={self.ph_X: Xn, self.ph_dA: dAn, self.ph_YLabel: YLabeln,
                                                self.ph_Y_prestep: Y_prestepn, self.ph_gradY_prestep: gradY_prestepn})

        #### Main Optimization
        for step in range(n_totalstep):
            if step < n_unitstep or step % n_unitstep == 0 or step == n_totalstep - 1:
                glstep, lr, br, b, gradY, regloss, _ = \
                    sess.run([self.step, self.rate, self.bn_rate, self.ensemble_b, self.gradY, self.Loss, self.Optimizer])
                print(" step : ", "{:4d}".format(glstep),
                      ",  learning rate: ", "{0:.6f}".format(lr),
                      ",  batch learning rate: ", "{0:.6f}".format(br),
                      ",  b : ", "{0:.6f}".format(b),
                      ",  gradY : ", "{0:.6f}".format(gradY.mean()),
                      ",  regloss : ", "{0:.6f}".format(regloss * 10000))
            else:
                sess.run(self.Optimizer)

    def train(self, sess, n, AmerOp, n_totalstep, n_unitstep, n_relaxstep, n_decaystep,
              init_rate, decay_rate):
        '''
        Implement the proposed American option algorithm for the n-th timestep
        '''
        print("\n **** ", self.NN_name, ", Training : ")
        self.n = n
        sess.run([tf.assign(self.delta, updaten - self.n % updaten),
                  tf.assign(self.init_rate, init_rate), tf.assign(self.decay_rate, decay_rate),
                  tf.assign(self.n_relaxstep, n_relaxstep), tf.assign(self.n_decaystep, n_decaystep)])

        def customer_reshape(Xn):
            if Xn.ndim == 2:
                return Xn.reshape((len(Xn) // num_channels, num_channels, Xn.shape[1]))
            if Xn.ndim == 3:
                return Xn.reshape((len(Xn) // num_channels, num_channels, Xn.shape[1], Xn.shape[2]))

        #### Data input
        np.random.shuffle(simulation_index)
        Xn = customer_reshape(AmerOp.X.values[simulation_index, self.n])
        dAn = customer_reshape(AmerOp.X.values[simulation_index, self.n + 1]) - np.exp(mu * dt) * Xn
        YLabeln = customer_reshape(AmerOp.YLabel.values[simulation_index, self.n])
        Y_prestepn = customer_reshape(AmerOp.Y.values[simulation_index, self.n])
        gradY_prestepn = customer_reshape(AmerOp.gradY.values[simulation_index, self.n])

        sess.run(self.test_init_op, feed_dict={self.ph_X: Xn, self.ph_Y_prestep: Y_prestepn,
                                               self.ph_gradY_prestep: gradY_prestepn})
        sess.run(self.input_mv_init_op)

        #### Main Optimization
        t0 = time.time()
        self.fit_Optimizer(sess, Xn, dAn, YLabeln, Y_prestepn, gradY_prestepn, n_totalstep, n_unitstep)
        print("\n   time: ", "{0:.2f}".format(time.time() - t0))

    def predict_Y_gradY_timestep(self, sess, AmerOp, m):
        assert self.n < N
        print("\n Ensemble Average : ", str(m))
        t0 = time.time()
        sess.run(self.test_init_op, feed_dict={
            self.ph_X: AmerOp.X.values[:, [m]], self.ph_Y_prestep: AmerOp.Y.values[:, [m]],
            self.ph_gradY_prestep: AmerOp.gradY.values[:, [m]]})
        AmerOp.Y.values[:, [m]], AmerOp.gradY.values[:, [m]] = sess.run([self.ensemble_Y, self.ensemble_gradY])
        # assert ensemble_Y.shape == (simulation_size, 1, 1)
        # assert ensemble_gradY.shape == (simulation_size, 1, d, 1)
        print(" time: " + "{0:.2f}".format(time.time() - t0))

    def predict(self, sess, AmerOp):
        t0 = time.time()
        if self.n % updaten == 0:
            for m in range(1, self.n + 1):
                self.predict_Y_gradY_timestep(sess=sess, AmerOp=AmerOp, m=m)
        else:
            self.predict_Y_gradY_timestep(sess=sess, AmerOp=AmerOp, m=self.n)

