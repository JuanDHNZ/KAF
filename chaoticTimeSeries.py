# -- coding: utf-8 --   
"""
Created on Tue Jun 30 11:51:26 2020

@author: Juan David

Atractores obtenidos del repositorio https://github.com/capitanov/chaospy
"""
import numpy as np

def chua(x=0, y=0, z=1, **kwargs):
    """
    Calculate the next coordinate X, Y, Z for Chua system.

    Parameters
    ----------
    x, y, z : float
        Input coordinates Z, Y, Z respectively
    kwargs : float
        alpha, beta, mu0, mu1 - are Chua system parameters
    """
    # Default parameters:
    alpha = kwargs.get('alpha', 15.6)
    beta = kwargs.get('beta', 28)
    mu0 = kwargs.get('mu0', -1.143)
    mu1 = kwargs.get('mu1', -0.714)

    ht = mu1*x + 0.5*(mu0 - mu1)*(np.abs(x + 1) - np.abs(x - 1))
    # Next step coordinates:
    # Eq. 1:
    x_out = alpha*(y - x - ht)
    y_out = x - y + z
    z_out = -beta*y
    # Eq. 2:
    # x_out = 0.3*y + x - x**3
    # y_out = x + z
    # z_out = y

    return x_out, y_out, z_out

def lorenz(x=0, y=0, z=0, **kwargs):
    """
    Calculate the next coordinate X, Y, Z for 3rd-order Lorenz system

    Parameters
    ----------
    x, y, z : float
        Input coordinates Z, Y, Z respectively
    kwargs : float
        beta, rho and sigma - are Lorenz system parameters

    """
    # Default Lorenz parameters:
    sigma = kwargs.get('sigma', 10)
    beta = kwargs.get('beta', 8/3)
    rho = kwargs.get('rho', 28)

    # Next step coordinates:
    x_out = sigma * (y - x)
    y_out = rho*x - y - x*z
    z_out = x*y - beta*z

    return x_out, y_out, z_out

def lotka_volterra(x=0, y=0, z=0):
    """
    Calculate the next coordinate X, Y, Z for Lotka–Volterra

    Parameters
    ----------
    x, y, z : float
        Input coordinates Z, Y, Z respectively
    """
    # Next step coordinates:
    x_out = x*(1 - x - 9*y)
    y_out = -y*(1 - 6*x - y + 9*z)
    z_out = z*(1 - 3*x - z)
    return x_out, y_out, z_out


def duffing(x=0, y=0, z=0, **kwargs):
    """
    Calculate the next coordinate X, Y, Z for Duffing map.
    It is 2nd order attractor (Z coordinate = 1)

    Duffing map:
    Eq. 1:
        dx/dt = y
        dy/dt = -a*y - x**3 + b * cos(z)
        dz/dt = 1
    where a = 0.1 and b = 11 (default parameters)

    Eq. 2:
        dx/dt = y
        dy/dt = a*y - y**3 - b*x
        dz/dt = 1
    where a = 2.75 and b = 0.2 (default parameters)

    Parameters
    ----------
    x, y, z : float
        Input coordinates Z, Y, Z respectively
    kwargs : float
        a and b - are Duffing system parameters
    """
    # Default parameters:
    # a = kwargs.get('a', 2.75)
    # b = kwargs.get('b', 0.2)
    a = kwargs.get('a', 0.1)
    b = kwargs.get('b', 11)

    # Next step coordinates:
    x_out = y
    y_out = -a*y - x**3 + b*np.cos(z)
    # y_out = a*y - y**3 - b*x
    z_out = 1
    return x_out, y_out, z_out



def nose_hoover(x=0, y=0, z=0):
    """
    Calculate the next coordinate X, Y, Z for 3rd-order Nose-Hoover

    Parameters
    ----------
    x, y, z : float
        Input coordinates Z, Y, Z respectively
    """
    # Next step coordinates:
    x_out = y
    y_out = y * z - x
    z_out = 1 - y * y

    return x_out, y_out, z_out


def rikitake(x=0, y=0, z=0, **kwargs):
    """
    Calculate the next coordinate X, Y, Z for 3rd-order Rikitake system

    Parameters
    ----------
    x, y, z : float
        Input coordinates Z, Y, Z respectively
    kwargs : float
        mu, a - are Rikitake system parameters

    """
    # Default Rikitake parameters:
    a = kwargs.get('a', 5)
    mu = kwargs.get('mu', 2)

    # Next step coordinates:
    x_out = -mu * x + z * y
    y_out = -mu * y + x * (z - a)
    z_out = 1 - x * y

    return x_out, y_out, z_out


def rossler(x=0, y=0, z=0, **kwargs):
    """
    Calculate the next coordinate X, Y, Z for 3rd-order Rossler system

    Parameters
    ----------
    x, y, z : float
        Input coordinates Z, Y, Z respectively
    kwargs : float
        a, b and c - are Rossler system parameters

    """
    # Default Rossler parameters:
    a = kwargs.get('a', 0.2)
    b = kwargs.get('b', 0.2)
    c = kwargs.get('c', 5.7)

    # Next step coordinates:
    x_out = -(y + z)
    y_out = x + a * y
    z_out = b + z * (x - c)

    return x_out, y_out, z_out


def wang(x=0, y=0, z=0):
    """
    Calculate the next coordinate X, Y, Z for 3rd-order Wang Attractor

    Parameters
    ----------
    x, y, z : float
        Input coordinates Z, Y, Z respectively
    """
    # Next step coordinates:
    x_out = x - y*z
    y_out = x - y + x*z
    z_out = -3*z + x*y
    return x_out, y_out, z_out