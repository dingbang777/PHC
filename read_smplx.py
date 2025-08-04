import numpy as np
import smplx
import trimesh
import torch
amass_data = np.load("/data-local/dingbang/phys_hoi_recon/PHC/ACCAD/Male1Walking_c3d/Walk_B10_-_Walk_turn_left_45_stageii.npz", allow_pickle=True)
#convert np to dict
smpl_data_entry = amass_data
print(smpl_data_entry.keys())
B = smpl_data_entry['poses'].shape[0]

start, end = 0, 0

pose_aa = smpl_data_entry['poses'].copy()[start:][:,:156]
print(pose_aa.shape, pose_aa.dtype)
root_trans = smpl_data_entry['trans'].copy()[start:]


beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy()
if len(beta.shape) == 2:
    beta = beta[0]
pose_aa = torch.from_numpy(pose_aa).float()
if isinstance(beta, np.ndarray):
    beta = torch.from_numpy(beta).float().unsqueeze(0).expand(B, -1)
root_trans = torch.from_numpy(root_trans).float()
# output_path = '.'
smplh = smplx.create('/data-local/dingbang/phys_hoi_recon/InterAct/models', model_type='smplh', gender='male', use_pca=False)
output = smplh(global_orient=pose_aa[:, :3],
                body_pose=pose_aa[:, 3:66],
                left_hand_pose=pose_aa[:, 66:111],
                right_hand_pose=pose_aa[:, 111:156],
                betas=beta,
                transl=root_trans,
                return_verts=True)
vertices = output.vertices.detach().cpu().numpy()
joints = output.joints.detach().cpu().numpy()

# for i in range(vertices.shape[0]):
mesh = trimesh.Trimesh(vertices=vertices[0], faces=smplh.faces)
mesh.export(f'mesh_amass.obj')