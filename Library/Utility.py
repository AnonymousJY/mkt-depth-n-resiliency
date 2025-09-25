#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


UST_TENOR_MAP = {
    '1m': np.array(1/12),
    '1.5m': np.array(1.5/12),
    '2m': np.array(2/12),
    '3m': np.array(3/12),
    '4m': np.array(4/12),
    '6m': np.array(6/12),
    '1y': np.array(1.),
    '2y': np.array(2.),
    '3y': np.array(3.),
    '5y': np.array(5.),
    '7y': np.array(7.),
    '10y': np.array(10.),
    '20y': np.array(20.),
    '30y': np.array(30.)
}
