'''
Main Script of Using Neural Network for solving American Option Pricing and Hedging.
The algorithm was proposed in https://arxiv.org/abs/1909.11532
'''

from amer_op_neural_net.config import *
from amer_op_neural_net.amer_op import AmericanOption
from amer_op_neural_net.neural_net import NeuralNet

## Main Script: American Option
AmerOp = AmericanOption(sample_size=simulation_size)

## Main Script: Neural Network
start_time = time.time()

print("\n\n************** Main Script: Neural Network **************\n")
dim_layers = [None, d + 5, d + 5, d + 5, d + 5, d + 5, d + 5, d + 5, 1]
NN = NeuralNet(NN_name="AmerOp", dim_layers=dim_layers)

## Main Script: Neural Network
with tf.Session(config=session_config) as sess:
    tf.summary.FileWriter("tf_graph", sess.graph)
    for n in reversed(range(N)):
        print("\n\n************** n = ", n, " **************\n")
        print("\n **** n = ", n, ", Pre-processing ")
        AmerOp.initialize_price_timestep(n)

        if n > 0:
            print("\n **** n = ", n, ", Neural Network ")
            if n == N - 1:
                sess.run(tf.global_variables_initializer())
                NN.train(sess, n, AmerOp, n_totalstep=n_totalstep, n_unitstep=20, n_relaxstep=n_relaxstep, n_decaystep=n_decaystep,
                         init_rate=0.01, decay_rate=0.01)

            sess.run(tf.variables_initializer([
                Var for Var in tf.global_variables()
                if "Adam" in Var.name or "power" in Var.name or "step" in Var.name]))
            NN.train(sess, n, AmerOp, n_totalstep=n_totalstep, n_unitstep=20, n_relaxstep=n_relaxstep, n_decaystep=n_decaystep,
                     init_rate=0.01, decay_rate=0.001)

            NN.predict(sess, AmerOp)
            AmerOp.update_hedging(n=n)
            AmerOp.evaluate_control_timestep(n, control_tol=control_tol)

            print("\n **** n = ", n, ", Post-processing ")
            AmerOp.update_price_timestep(n, nu=0.5)
            AmerOp.update_stop(n=n)
        else:
            assert n == 0
            print("\n **** n = ", n, ", Evaluation ")
            AmerOp.evaluate_at_initial_timestep()

        AmerOp.evaluate_quality(n)

print("\nTotal running time : ", "{0:.2f}".format(time.time() - start_time))

## Main Script: Neural Network
AmerOp.evaluate_results_final()

## Main Script: Hedging
AmerOp.delta_hedging()
AmerOp.evaluate_delta_hedging()

