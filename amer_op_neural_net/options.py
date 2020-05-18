'''
Collections of the common functions in option problems
'''

from .config import *


def __chainrule(gradZ1, gradZ2):
    assert 'array' in gradZ1.__class__.__name__ or 'Tensor' in gradZ1.__class__.__name__
    if 'array' in gradZ1.__class__.__name__:  # numpy.ndarray
        return np.expand_dims(gradZ1, axis=-2) * gradZ2
    elif 'Tensor' in gradZ1.__class__.__name__:  # tf.Tensor
        return tf.expand_dims(gradZ1, axis=-2) * gradZ2


def func_digital(X, sharpness=None):
    assert 'array' in X.__class__.__name__ or 'Tensor' in X.__class__.__name__
    if 'array' in X.__class__.__name__:  # numpy.ndarray
        if sharpness is None:
            return (X > 0.0).astype(np_floattype)
        else:
            def sigmoid(x):
                # return 1 / (1 + np.exp(-x))
                y = np.maximum(x, 0.0)
                ex = np.exp(x - y)
                ey = np.exp(-y)
                return ex / (ex + ey)

            return sigmoid(sharpness * X)
    elif 'Tensor' in X.__class__.__name__:  # tf.Tensor
        if sharpness is None:
            return tf.cast(X > 0.0, tf_floattype)
        else:
            return tf.nn.sigmoid(sharpness * X)


def func_relu(X, sharpness=None):
    assert 'array' in X.__class__.__name__ or 'Tensor' in X.__class__.__name__
    if 'array' in X.__class__.__name__:  # numpy.ndarray
        if sharpness is None:
            return np.maximum(X, 0.0)
        else:
            def softplus(x):
                # return np.log(1 + np.exp(x))
                y = np.maximum(x, 0.0)
                return y + np.log(np.exp(x - y) + np.exp(-y))

            return 1 / sharpness * softplus(sharpness * X)
    elif 'Tensor' in X.__class__.__name__:  # tf.Tensor
        if sharpness is None:
            return tf.nn.relu(X)  # tf.maximum(X, 0.0)
        else:
            return 1 / sharpness * tf.nn.softplus(sharpness * X)


def func_max(X, sharpness=None):
    assert 'array' in X.__class__.__name__ or 'Tensor' in X.__class__.__name__
    if 'array' in X.__class__.__name__:  # numpy.ndarray
        if sharpness is None:
            return X.max(axis=-1, keepdims=True)
            # return np.amax(X, axis=-1, keepdims=True)
        else:
            def logsumexp(x):
                # return np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
                xmax = x.max(axis=-1, keepdims=True)
                return xmax + np.log(np.sum(np.exp(x - xmax), axis=-1, keepdims=True))

            return 1 / sharpness * logsumexp(sharpness * X)
    elif 'Tensor' in X.__class__.__name__:  # tf.Tensor
        if sharpness is None:
            return tf.reduce_max(X, axis=-1, keepdims=True)
        else:
            return 1 / sharpness * tf.reduce_logsumexp(sharpness * X, axis=-1, keepdims=True)


def func_grad_max(X, X_reduced, sharpness=None):
    assert 'array' in X.__class__.__name__ or 'Tensor' in X.__class__.__name__
    if 'array' in X.__class__.__name__:  # numpy.ndarray
        if sharpness is None:
            return (X > X_reduced - 1e-6).astype(np_floattype)
        else:
            def softmax(x):
                # ex = np.exp(x)
                xmax = x.max(axis=-1, keepdims=True)
                ex = np.exp(x - xmax)
                return ex / np.sum(ex, axis=-1, keepdims=True)

            return softmax(sharpness * X)
    elif 'Tensor' in X.__class__.__name__:  # tf.Tensor
        if sharpness is None:
            return tf.cast(X > X_reduced - 1e-6, tf_floattype)
        else:
            return tf.nn.softmax(sharpness * X, axis=-1)


def func_geometric_mean(X):
    assert 'array' in X.__class__.__name__ or 'Tensor' in X.__class__.__name__
    if 'array' in X.__class__.__name__:  # numpy.ndarray
        d = X.shape[-1]
        # return np.abs(np.prod(X, axis=-1, keepdims=True)) ** (1 / d)
        return np.prod(np.abs(X) ** (1 / d), axis=-1, keepdims=True)
    elif 'Tensor' in X.__class__.__name__:  # tf.Tensor
        d = tf.cast(tf.shape(X)[-1], tf_floattype)
        # return tf.abs(tf.reduce_prod(X, axis=-1, keepdims=True)) ** (1 / d)
        return tf.reduce_prod(tf.abs(X) ** (1 / d), axis=-1, keepdims=True)


def func_grad_geometric_mean(X, X_reduced):
    assert 'array' in X.__class__.__name__ or 'Tensor' in X.__class__.__name__
    if 'array' in X.__class__.__name__:  # numpy.ndarray
        d = X.shape[-1]
        return np.prod(np.sign(X), axis=-1, keepdims=True) * X_reduced / (d * (X + np.sign(X) * 1e-6))
    elif 'Tensor' in X.__class__.__name__:  # tf.Tensor
        d = tf.cast(tf.shape(X)[-1], tf_floattype)
        return tf.reduce_prod(tf.sign(X), axis=-1, keepdims=True) * X_reduced / (d * (X + tf.sign(X) * 1e-6))


def levelset_function(X, eval_grad=False, sharpness=None):
    '''
    Compute the "levelset" function L in the payoff function: P = max(L, 0)

    input:
    X:     Numpy array Tensorflow Tensor of the dimension [None, ..., None, d]
    output:
    L:     Numpy array Tensorflow Tensor of the dimension [None, ..., None, 1]
    gradL: Numpy array Tensorflow Tensor of the dimension [None, ..., None, d, 1]
    '''

    assert 'array' in X.__class__.__name__ or 'Tensor' in X.__class__.__name__
    assert 'call' in option_type or 'put' in option_type
    assert 'max' in option_type or 'arithmic' in option_type or 'geometric' in option_type
    if 'call' in option_type:
        sign = 1
    elif 'put' in option_type:
        sign = -1

    if 'max' in option_type:
        X_reduced = func_max(X, sharpness=sharpness)
    elif 'arithmic' in option_type:
        X_reduced = func_arithmic_mean(X)
    elif 'geometric' in option_type:
        X_reduced = func_geometric_mean(X)
    L = sign * (X_reduced - K)

    if eval_grad:
        if 'max' in option_type:
            gradX_reduced = func_grad_max(X, X_reduced, sharpness=sharpness)
        elif 'arithmic' in option_type:
            gradX_reduced = func_grad_arithmic_mean(X, X_reduced)
        elif 'geometric' in option_type:
            gradX_reduced = func_grad_geometric_mean(X, X_reduced)

        if 'array' in X.__class__.__name__:  # numpy.ndarray
            gradL = sign * np.expand_dims(gradX_reduced, axis=-1)
        elif 'Tensor' in X.__class__.__name__:  # tf.Tensor
            gradL = sign * tf.expand_dims(gradX_reduced, axis=-1)

        return (L, gradL)
    else:
        return L


def payoff_function(L, gradL=None, sharpness=None):
    '''
    Compute the payoff function: P = max(L, 0)

    input:
    L:     Numpy array Tensorflow Tensor of the dimension [None, ..., None, 1]
    gradL: Numpy array Tensorflow Tensor of the dimension [None, ..., None, d, 1]

    output:
    P:     Numpy array Tensorflow Tensor of the dimension [None, ..., None, 1]
    gradP: Numpy array Tensorflow Tensor of the dimension [None, ..., None, d, 1]
    '''

    assert 'vanilla' in option_type
    P = func_relu(L, sharpness)

    if gradL is None:
        return P
    else:
        gradP = func_digital(L, sharpness)
        gradP = __chainrule(gradP, gradL)
        return (P, gradP)
