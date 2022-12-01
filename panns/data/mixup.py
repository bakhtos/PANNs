import numpy as np

__all__ = ['mixup_coefficients',
           'mixup']


def mixup_coefficients(mixup_alpha: float, batch_size: int = 32,
                       random_seed: int = 1234) -> np.ndarray:
    """Sample lambda coefficients for mixup from beta distribution.

    Args:
        mixup_alpha: float, Will be used as both parameters for the beta distribution
        batch_size: int, Batch size to generate lambdas for;
            must be twice as big as the actual batch size passed to model
            (default 32)
        random_seed: int, Seed to give to numpy.random.default_rng
            (default 1234)

    Returns: numpy.ndarray, Each call yields a batch of lambda coefficients

    """

    random_state = np.random.default_rng(random_seed)
    while True:
        lambdas = np.zeros(batch_size)
        lam = random_state.beta(mixup_alpha, mixup_alpha, batch_size//2)
        lambdas[0::2] = lam
        lambdas[1::2] = 1.0 - lam
        yield lambdas


def mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes (1, 3, 5, ...).

    Args:
        x: numpy.ndarray, Batch of data of shape (batch_size * 2, ...) to be mixed
        mixup_lambda: numpy.ndarray, Array of shape (batch_size * 2,),
            lambda coefficients for mixup

    Returns: numpy.ndarray, Array of shape (batch_size, ...) with performed mixup

    """

    out = x.transpose(-1, 0) * mixup_lambda
    out = out[::2]+out[1::2]
    return out.transpose(-1,0)
