# scil_flip_surface.py new/new_vtk_lh.gray.vtk flipped.vtk x y -f

import numpy as np
import nibabel as nib
import sys
img = nib.load(sys.argv[1]) #7T image as input
inv_aff = np.linalg.inv(img.affine)

rot = img.affine[0:3,0:3]
trans = np.dot(rot, inv_aff[0:3,3] - (np.array(img.shape) / 2))

new_aff = np.eye(4)
new_aff[0:3,3] = trans
new_aff[0:2,3] *= -1
np.savetxt(sys.argv[2], new_aff)
#scil_apply_transform_to_surface.py flipped.vtk tmp_aff.txt transfo.vtk -f
