import numpy as np

file_path = '/data-local/dingbang/phys_hoi_recon/PHC/ACCAD/Male1Walking_c3d/Walk_B4_-_Stand_to_Walk_Back_stageii.npz'

data = np.load(file_path, allow_pickle=True)

print("Keys in the npz file:")
print(data.keys())

for key in data.keys():
    print(f"\nKey: {key}")
    if isinstance(data[key], np.ndarray):
        print(f"Shape: {data[key].shape}, Dtype: {data[key].dtype}")
    elif isinstance(data[key], list):
        print(f"Length: {len(data[key])}, Type: {type(data[key])}")
    else:
        print(f"Type: {type(data[key])}")
    # print(f"Value: {data[key]}")

print(data['poses'].shape)