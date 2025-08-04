import torch
import joblib
import numpy as np
# Specify the file path
# file_path = '/data-local/dingbang/phys_hoi_recon/InterAct/data/intercap/sequences/Sub01_Object01_Seg_1_suitcase/Sub01_Object01_Seg_1_suitcase.pt'

# # Load the PyTorch file
# data = torch.load(file_path)

# # If you need to check the contents
# print("Loaded data type:", type(data))


#load from vis_motion_hoi
file_path2 = '/data-local/dingbang/phys_hoi_recon/PHC/intercap2.pt'
data2 = torch.load(file_path2)

obj_data = joblib.load('intercap_obj_test2.pkl')
obj_trans = torch.from_numpy(obj_data['trans']).float()
obj_rot = torch.from_numpy(obj_data['rot']).float()

# Print info about second file
print("Loaded data2 type:", type(data2))
print("Keys in data2:", data2.keys())
bsz = data2['root_pos'].squeeze(1).shape[0]
data = torch.zeros((bsz, 591), dtype=torch.float32)
data[:,:3] = data2['root_pos'].squeeze(1)
data[:,3:7] = data2['root_rot'].squeeze(1)
data[:,9:9+153] = data2['dof_pos'].squeeze(1)
print("Loaded data2 shape111111:", data2['rg_pos'].shape)
data[:,162:162+52*3] = data2['rg_pos'].squeeze(1).reshape(bsz, 52*3)
data[:, 331+52:331+52+52*4] = data2['rb_rot'].squeeze(1).reshape(bsz, 52*4)

data[:,318:318+3] = obj_trans
data[:,321:321+4] = obj_rot


from contact import get_contact
human_path = '/data-local/dingbang/phys_hoi_recon/InterAct/data/omomo/sequences_canonical/sub8_smallbox_002/human.npz'
obj_path = '/data-local/dingbang/phys_hoi_recon/InterAct/data/omomo/sequences_canonical/sub8_smallbox_002/object.npz'
obj_verts = '/data-local/dingbang/phys_hoi_recon/InterAct/data/omomo/objects/smallbox/smallbox.obj'
human_contact, obj_contact = get_contact(human_path, obj_path, obj_verts)
data[:,330:331] = obj_contact
data[:, 331:331+52] = human_contact

torch.save(data,'/data-local/dingbang/phys_hoi_recon/InterMimic/InterAct/InterCap/sub8_smallbox_002.pt')

