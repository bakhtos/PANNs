import numpy as np

__all__ = ['mixup_coefficients',
           'do_mixup']

def mixup_coefficients(mixup_alpha, random_seed=1234, batch_size=32):
    random_state = np.random.default_rng(random_seed)
    while True:
        lambdas = np.zeros(batch_size)
        lam = random_state.beta(mixup_alpha, mixup_alpha, batch_size//2)
        lambdas[0::2] = lam
        lambdas[1::2] = 1.0 - lam
        yield lambdas

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes (1, 3, 5, ...).

      :param x: array of shape (batch_size * 2, ...) to be mixed
      :type x: numpy.ndarray
      :param mixup_lambda: array of shape (batch_size * 2,), coefficients for mixup
      :type mixup_lambda: numpy.ndarray 
      :return out: Array of shape (batch_size, ...) with performed mixup
      :rtype: numpy.ndarray
    """

    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out
