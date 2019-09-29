import numpy as np

def barron_loss(x, alpha=1, c=0.01):
    t1 = np.abs(alpha-2) / alpha # term 1
    base = ((x/c)**2) / np.abs(alpha-2) + 1
    loss = t1 * (np.power(base, alpha/2) - 1)
    return loss

def DSSIM(I_x, I_y,
          C1=0.01 ** 2,
          C2=0.03 ** 2):
    """Given two windows of the same shape, calculate DSSIM"""
    mu_x = np.mean(I_x)
    mu_y = np.mean(I_y)
    sigma_x  = np.mean(I_x ** 2) - mu_x ** 2
    sigma_y  = np.mean(I_y ** 2) - mu_y ** 2
    sigma_xy = np.mean(I_x*I_y) - mu_x * mu_y

    numer = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denom = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return np.clip((1 - numer/denom)/2, a_min=0, a_max=1)

def loss(alpha=0.85):
    """calculate the loss, where alpha is the weight of the DSSIM term"""
    pass
