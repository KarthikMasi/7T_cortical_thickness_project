# 7T_cortical_thickness_project
```
usage: high_res_angles_extraction.py [-h] --img IMG_FN --surface SURF_FN --out
                                     OUT --angle_np ANGLES --z Z_DIST

optional arguments:
  -h, --help         show this help message and exit
  --img IMG_FN       Path of 7T SWI image to register
  --surface SURF_FN  Path of surface registered to 7T space
  --out OUT          filename of image to be created with heatmap of angles
  --angle_np ANGLES  filename to write numpy array of normal angles with z
                     vector to txt
  --z Z_DIST         filename to write numpy array of z distance coordinates
                     to txt
```

## Preprocessing

```
scil_flip_surface.py surf.vtk flipped.vtk x y -f

python get_xform_transfo.py 3T_t1.nii.gz transform.txt

scil_apply_transform_to_surface.py flipped.vtk tranform.txt flipped.3t.vtk 
```


###ipython live code:
```
import ants 

fixed_image = ants.image_read('7t_swi.nii.gz') 

moving_image = ants.image_read('3T_t1.nii.gz') 

transform = ants.registration(fixed_image,moving_image,'Affine', verbose=True) 

#manual copy of fwdtransform file to code location 

cp /tmp/tmpvvyilq030GenericAffine.mat 7t_aff.mat 
```
from ants bin folder 

```
./ConvertTransformFile 3 7t_aff.mat 7t_aff.txt --hm 

scil_apply_transform_to_surface.py flipped.3t.vtk 7t_aff.txt surf_7t.vtk 
```
