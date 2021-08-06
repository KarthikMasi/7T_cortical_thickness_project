# 7T_cortical_thickness_project

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
