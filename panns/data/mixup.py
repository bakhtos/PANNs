import numpy as np

__all__ = ['Mixup',
           'do_mixup']

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator."""

        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Mixup random coefficients.

           :param int batch_size: batch size of the dataset
           :return mixup_lambdas: Array of lambda coefficients, with shape (batch_size,)
           :rtype: numpy.ndarray
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)

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
