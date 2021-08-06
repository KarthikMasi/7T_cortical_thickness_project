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
    print("Loading img and vtk object...")
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

def make_vox_normal_vector(normals,points,img,z_fn):
    """
    Converts vtk format coordinates(mm) to voxel space and returns unit normal vectors
    """
    print("Making unit normal vector with normal and surface points...")
    normals_surface_np = points + normals
    normals_vox = trimeshpy.vtk_util.vtk_to_vox(normals_surface_np,img)
    surface_vox = trimeshpy.vtk_util.vtk_to_vox(points,img)
    normal_vector_vox = np.zeros(normals_vox.shape)
    for i in range(len(normal_vector_vox)):
        normal_vector_vox[i] = (normals_vox[i] - surface_vox[i])/np.linalg.norm(normals_vox[i] - surface_vox[i])
    np.savetxt(z_fn,normal_vector_vox[:,2])
    return normals_vox,surface_vox,normal_vector_vox

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
        z_normal_angles[i] = math.acos(np.dot(normal_vector[i],z_vector[i])) * 180 / math.pi
    return z_normal_angles

def make_image_with_angles(surface_vox,dot_product_angles,out,img,angles):
    """
    Writes a nii file with the value range governed by the dot product angles
    """
    print("Writing to image...")
    angled_image = np.zeros(img.shape)
    dot_product_normalized = np.zeros(dot_product_angles.shape)
    for i in range(len(dot_product_angles)):
        if dot_product_angles[i] <= 90.0:
            dot_product_normalized[i] = dot_product_angles[i]
        else:
            dot_product_normalized[i] = 180 - dot_product_angles[i]

    np.savetxt(angles,dot_product_normalized)
    for i in range(len(surface_vox)):
        if surface_vox[i,2] <= 79:
            j = surface_vox[i].astype(int)
            indices = (tuple([int(j[0]),int(j[1]),int(j[2])]))
            angled_image[indices] = dot_product_normalized[i]
    nib.save(nib.Nifti1Image(angled_image,img.affine),out)

def extract(args):
    """
    Main code block
    """
    img, surf = load_img_n_surface(args.img_fn,args.surf_fn)
    tesselated_surf = tesselate_surface(surf,img)
    normals, points = compute_normal(tesselated_surf)
    normal_voxels,surface_voxels,normal_vector = make_vox_normal_vector(normals,points,img,args.z_dist)
    dot_product_angles = compute_dot_product_with_z(normal_vector)
    make_image_with_angles(surface_voxels,dot_product_angles,args.out,img,args.angles)

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
    parser.add_argument("--angle_np",dest='angles',default=None,required=True,\
                        help="filename to write numpy array of normal angles with z vector to txt")
    parser.add_argument("--z",dest="z_dist",default=None,required=True,\
                        help="filename to write numpy array of z distance coordinates to txt")
    return parser

if __name__== '__main__':
    parser = add_to_parser()
    OPTIONS = parser.parse_args()
    extract(OPTIONS)

