import tensorflow as tf


# ************************************************************************* #
#                            learning rate                                  #
# ************************************************************************* #
def constant(global_step, **kwargs):

    lr = tf.constant(kwargs['lr'], name='learning_rate')

    return lr

def exponential_decay(global_step, **kwargs):

    lr = tf.train.exponential_decay(kwargs['lr'],global_step, kwargs['decay_steps'],
                                    kwargs['decay_rate'],kwargs['staircase'],name='learning_rate')

    return lr

def piecewise_constant(global_step, **kwargs):

    lr = tf.train.piecewise_constant(global_step, kwargs['boundaries'], kwargs['values'], name='learning_rate')

    return lr

def polynomial_decay(global_step, **kwargs):

    lr = tf.train.polynomial_decay(kwargs['lr'], global_step, kwargs['decay_steps'],
                                   kwargs['end_learning_rate'], kwargs['power'], kwargs['cycle'], name='learning_rate')

    return lr

lr_map = {
    'constant':constant,
    'exp': exponential_decay,
    'piecewise': piecewise_constant,
    'polynominal':polynomial_decay,
}

def get_lr(name, global_step, **kwargs):

    if name not in lr_map:
        raise ValueError('Name of learning rate mode:{} unknown'.format(name))

    return lr_map[name](global_step, **kwargs)


# ************************************************************************* #
#                             optimizer                                     #
# ************************************************************************* #

def sgd(lr, **kwargs):
    return tf.train.GradientDescentOptimizer(lr)

def mom(lr, **kwargs):
    return tf.train.MomentumOptimizer(lr, kwargs['momentum'])

def rms(lr, **kwargs):
    return tf.train.RMSPropOptimizer(lr, kwargs['decay'], kwargs['momentum'])

def adam(lr, **kwargs):
    return tf.train.AdamOptimizer(lr, kwargs['belta1'], kwargs['belta2'])

def adagrad(lr, **kwargs):
    return tf.train.AdagradOptimizer(lr, kwargs['initial_accumulator_value'])

def adadelta(lr, **kwargs):
    return tf.train.AdadeltaOptimizer(lr, kwargs['rho'])

optim_map = {
    'sgd':sgd,
    'mom':mom,
    'rms':rms,
    'adam':adam,
    'adagrad':adagrad,
    'adadelta':adadelta,
}

def get_optim(name, lr, **kwargs):

    if not name in optim_map:
        raise ValueError('Name of Optimizer:{} unknown'.format(name))

    return optim_map[name](lr, **kwargs)