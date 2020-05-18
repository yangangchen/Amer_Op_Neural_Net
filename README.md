## Deep Neural Network Framework Based on Backward Stochastic Differential Equations for Pricing and Hedging American Options in High Dimensions: Source Code (CPU version)

Copyright (C) 2020 Yangang Chen, Justin W. L. Wan

https://arxiv.org/abs/1909.11532

#### Description

We propose a deep neural network framework for computing prices and deltas of American options in high dimensions. The architecture of the framework is a sequence of neural networks, where each network learns the difference of the price functions between adjacent timesteps. We introduce the least squares residual of the associated backward stochastic differential equation as the loss function. Our proposed framework yields prices and deltas on the entire spacetime, not only at a given point. The computational cost of the proposed approach is quadratic in dimension, which addresses the curse of dimensionality issue that state-of-the-art approaches suffer. Our numerical simulations demonstrate these contributions, and show that the proposed neural network framework outperforms state-of-the-art approaches in high dimensions.

#### Scripts

* **main.py**: the main script of using neural network for solving American option pricing and hedging.
* **amer_op_neural_net**: the package of using neural network for solving American option pricing and hedging.

#### Usage

* The input parameter of the algorithm is set in **amer_op_neural_net/config.py**
* To run the code, execute:

```
python ./main.py
```
