# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:10:36 2023

@author: alejandrs
"""
def comparar(a,b):
    vect=[]
    for i in a:
        for j in b:
            if i==j:
                vect.append(i)
    return vect