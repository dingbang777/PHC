import pickle

pkl_path = '/data-local/dingbang/phys_hoi_recon/PHC/sample_data/amass_copycat_occlusion_v3.pkl'

import numpy as np

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
except pickle.UnpicklingError:
    # Try loading as numpy file
    data = np.load(pkl_path, allow_pickle=True)

print(type(data))
print(data.keys() if hasattr(data, 'keys') else None)