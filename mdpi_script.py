# This is a script

import os
import numpy as np


def do_calculation(x, y, z):
    """
    Perform a multiplication operation on three numbers.

    Parameters:
    x (int or float): The first number to be multiplied.
    y (int or float): The second number to be multiplied.
    z (int or float): The third number to be multiplied.

    Returns:
    int or float: The product of x, y, and z.
    """
    print("I am doing a calculation. BANANAS! " + str(z))
    return x * y * z

if __name__ == "__main__":
    print("This is a script")
    x = 5
    y = 10
    print(do_calculation(x,y))
    print("Script is done")