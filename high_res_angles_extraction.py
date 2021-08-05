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

def tesseltate_surface(surf,img):
    """
    Tesselates surface such that maximum edge length is < least img voxel resolution side
    """
    subdivider = vtk.vtkAdaptiveSubdivisionFilter()
    subdivider.SetInputConnection(surf.GetOutputPort())
    subdivider.SetMaximumEdgeLength(min(img.header.get_zooms()-0.01))
    subdivider.Update()
    return subdivider

def compute_normal(surf):
    """
    Computes normal of vtk polydata and returns the normals and surface points as np array
    """
    normal_polydata = vtk.vtkPolyDataNormals()
    normal_polydata.SetInputConnection(surf.GetOutputPort())
    normal_polydata.ComputePointNormalsOn()
    normal_polydata.Update()
    pointdata = normal_polydata.GetOutput().GetPointData()
    normals = pointdata.GetNormals()
    points = normal_polydata.GetOutput().GetPoints()
    normals_np = vtk.util.numpy_support.vtk_to_numpy(normals)
    points_np = vtk.util.numpy_support.vtk_to_numpy(points.GetData())
    return normals_np,points_np

def extract(args):
    """
    Main code block
    """
    img, surf = load_img_n_surface(args.img_fn,args.surf_fn)
    tesselated_surf = tesselate_surface(surf,img)
    normals, points = compute_normal(surf)

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

