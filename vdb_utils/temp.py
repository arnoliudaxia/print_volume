import numpy as np
import cv2
import h5py
array_path = '/media/vrlab/rabbit/print3dingp/workspace/bunny_cloud_white/array/00000120.npy'

array = np.load(array_path)

print(array.shape)

print(array[:, :, 0].min())
print(array[:, :, 0].max())

print(array[:, :, 3].min())
print(array[:, :, 3].max())

cv2.imwrite('temp.png', array[:, :, 3]*40)


h5_file_path = 'vdb_utils/bunny_cloud.h5'

h5_file = h5py.File(h5_file_path, 'r')
h5_density = h5_file['density'][:]
cv2.imwrite('temp2.png', h5_density[100, :, :]*200)