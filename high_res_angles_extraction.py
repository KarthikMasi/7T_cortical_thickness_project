#!/usr/bin/env python
# -*- coding: utf-8 -*-

import vtk
import numpy as np
import nibabel as nib
import trimeshpy
import math
import argparse

def load_img_n_surface(img_fn,surf_fn):
    """
    Loads the image with nibabel. Loads the surface with vtkPolyDataReader
    """
    img = nib.load(img_fn)
    vtkreader = vtk.vtkPolyDataReader()
    vtkreader.SetFileName(surf_fn)
    vtkreader.Update()
    return img, vtkreader

def extract(args):
    """
    Main code block
    """
    img, surf = load_img_n_surface(args.img_fn,args.surf_fn)


def add_to_parser():
    """
    Arguments reader
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",dest=img_fn, default=None,required=True, \
                        help="Path of 7T SWI image to register")
    parser.add_argument("--surface",dest=surf_fn,default=None,required=True,\
                        help="Path of surface registered to 7T space")
    return parser

if __name__== '__main__':
    parser = add_to_parser()
    OPTIONS = parser.parse_args()
    extract(OPTIONS)

