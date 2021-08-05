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
    print("Loading img and vtk object")
    img = nib.load(img_fn)
    vtkreader = vtk.vtkPolyDataReader()
    vtkreader.SetFileName(surf_fn)
    vtkreader.Update()
    return img, vtkreader

def tesselate_surface(surf,img):
    """
    Tesselates surface such that maximum edge length is < least img voxel resolution side
    """
    print("Running tesselation....")
    subdivider = vtk.vtkAdaptiveSubdivisionFilter()
    subdivider.SetInputConnection(surf.GetOutputPort())
    subdivider.SetMaximumEdgeLength(min(img.header.get_zooms())-0.01)
    subdivider.Update()
    print("Tesselation complete....")
    return subdivider

def compute_normal(surf):
    """
    Computes normal of vtk polydata and returns the normals and surface points as np array
    """
    print("Computing normals...")
    normal_polydata = vtk.vtkPolyDataNormals()
    normal_polydata.SetInputConnection(surf.GetOutputPort())
    normal_polydata.ComputePointNormalsOn()
    normal_polydata.Update()
    pointdata = normal_polydata.GetOutput().GetPointData()
    normals = pointdata.GetNormals()
    points = normal_polydata.GetOutput().GetPoints()
    normals_np = vtk.util.numpy_support.vtk_to_numpy(normals)
    points_np = vtk.util.numpy_support.vtk_to_numpy(points.GetData())
    print("Normals ready in numpy format...")
    return normals_np,points_np

def make_vox_normal_vector(normals,points,img):
    """
    Converts vtk format coordinates(mm) to voxel space and returns unit normal vectors
    """
    print("Making unit normal vector with normal and surface points")
    normals_surface_np = points + normals
    normals_vox = trimeshpy.vtk_util.vtk_to_vox(normals_surface_np,img)
    surface_vox = trimeshpy.vtk_util.vtk_to_vox(points,img)
    print(normals_vox[0])
    print(surface_vox[0])
    normal_vector_vox = np.zeros(normals_vox.shape)
    for i in range(len(normal_vector_vox)):
        normal_vector_vox[i] = \
          (normals_vox[i] - surface_vox[i])/np.linalg.norm(normals_vox[i]- surface_vox[i])
    np.savetxt('/home/local/VANDERBILT/ramadak/3T_7T/z_vector_normal.txt',normal_vector_vox[:,2])
    return normal_vector_vox

def compute_dot_product_with_z(normal_vector):
    """
    Makes the unit z vectors and computes the angles
    """
    print("Computing dot product angles....")
    z_vector = np.zeros(normal_vector.shape)
    for i in range(len(normal_vector)):
        z_vector[i,2]=1
    z_normal_angles = np.zeros(len(normal_vector))
    for i in range(len(normal_vector)):
        z_normal_angles[i] = math.acos(np.dot(normal_vector[i],z_vector[i]))/ 180 * math.pi
    print(normal_vector[0])
    print(z_normal_angles.max())
    np.savetxt('/home/local/VANDERBILT/ramadak/3T_7T/angles.txt',z_normal_angles)
    return z_normal_angles

def make_image_with_angles(points,dot_product_angles,out,img):
    """
    Writes a nii file with the value range governed by the dot product angles
    """
    print("Writing to image...")
    angled_image = np.zeros(img.shape)
    points_int = points.astype(int)
    for i in range(len(points)):
        if points_int[i,2] <= 79:
            indices = points_int[i]
            angled_image[indices[0],indices[1],indices[2]] += dot_product_angles[i]
    nib.save(nib.Nifti1Image(angled_image,img.affine),out)
    print("Complete....")

def extract(args):
    """
    Main code block
    """
    img, surf = load_img_n_surface(args.img_fn,args.surf_fn)
    tesselated_surf = tesselate_surface(surf,img)
    normals, points = compute_normal(tesselated_surf)
    normal_vector = make_vox_normal_vector(normals,points,img)
    dot_product_angles = compute_dot_product_with_z(normal_vector)
    make_image_with_angles(points,dot_product_angles,args.out,img)


def add_to_parser():
    """
    Arguments reader
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",dest='img_fn', default=None,required=True, \
                        help="Path of 7T SWI image to register")
    parser.add_argument("--surface",dest='surf_fn',default=None,required=True,\
                        help="Path of surface registered to 7T space")
    parser.add_argument("--out",dest='out',default=None,required=True,\
                        help="filename of image to be created with heatmap of angles")
    return parser

if __name__== '__main__':
    parser = add_to_parser()
    OPTIONS = parser.parse_args()
    extract(OPTIONS)

