import numpy as np
import os
import os.path as osp

data_dir = 'data/blendedmvs_processed'
dir_list = os.listdir(data_dir)
dtype = [('seq_high', '<u4'),   # 32-bit unsigned integer (4 bytes)
         ('seq_low', '<u8'),    # 64-bit unsigned integer (8 bytes)
         ('img1', '<u4'),       # 32-bit unsigned integer (4 bytes)
         ('img2', '<u4'),       # 32-bit unsigned integer (4 bytes)
         ('score', '<f4')]      # 32-bit float (4 bytes)
data = []

for dir in dir_list:
    print(dir)
    if not os.path.isdir(os.path.join(data_dir, dir)):
        continue
    file_list = os.listdir(os.path.join(data_dir, dir))
    for file in file_list:
        if file.endswith('.jpg'):
            data.append((int(dir[:8],16),int(dir[8:],16), int(file[:-4]), int(file[:-4]), 1.0))
        
structured_array = np.array(data, dtype=dtype)

print(structured_array[:10])
for seqh, seql, img1, img2, score in structured_array[:10]:
        for view_index in [img1, img2]:
            impath = osp.join(data_dir, f"{seqh:08x}{seql:016x}", f"{view_index:08n}.jpg")
            assert osp.isfile(impath), f'missing image at {impath=}'
print(structured_array.dtype)
np.save('blendedmvs_single.npy', structured_array)
