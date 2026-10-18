#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:48:16 2020

@author: foleyj10
"""

import numpy as np
import matplotlib.pyplot as plt

n = 100
x = 1.*np.arange(n)
y = np.random.rand(n)
prop = x**2

fig = plt.figure(1, figsize=(5,5))
ax  = fig.add_subplot(111)
plot_colourline(x,y,prop)