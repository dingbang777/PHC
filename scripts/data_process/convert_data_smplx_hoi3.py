from ast import Try
import torch
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as sRot
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from smpl_sim.smpllib.smpl_mujoco_new import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import scipy.ndimage.filters as filters
from typing import List, Optional
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

robot_cfg = {
    "mesh": False,
    "model": "smplx",
    "rel_joint_lm": False,
    "upright_start": True,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}
print(robot_cfg)

smpl_local_robot = LocalRobot(
    robot_cfg,
    data_dir="data/smpl",
)

smpl_local_robot2 = LocalRobot(
    robot_cfg,
    data_dir="data/smpl",
)

# amass_data = joblib.load("/data-local/dingbang/phys_hoi_recon/PHC/ACCAD/Male1Walking_c3d/Walk_B10_-_Walk_turn_left_45_stageii.npz")
amass_data = np.load("/data-local/dingbang/phys_hoi_recon/InterAct/data/omomo/sequences_canonical/sub8_smallbox_030/human.npz", allow_pickle=True)
obj_data = np.load("/data-local/dingbang/phys_hoi_recon/InterAct/data/omomo/sequences_canonical/sub8_smallbox_030/object.npz", allow_pickle=True)
obj_data = obj_data
# convert np to dict
# amass_data = {k: v for k, v in amass_data.items()}
double = False

mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

 

amass_remove_data = []
obj_motion_dict = {}
# B = obj_data['angles'].shape[0]
print(obj_data.keys())


#axis transform for object
trans = obj_data['trans'].copy()
print('obj_bfore',trans[0])
pose_aa = obj_data['angles'].copy()
R1 = np.array([
    [ 1,  0,  0],
    [ 0,  0,  -1],
    [ 0,  1,  0]
])
R1 = sRot.from_matrix(R1)

pose_aa[:,0:3] = (R1 * sRot.from_rotvec(pose_aa[:,0:3])).as_rotvec()
pose_quat = sRot.from_rotvec(pose_aa).as_quat()
obj_motion_dict['rot'] = pose_quat
obj_motion_dict['trans'] = R1.apply(trans)
print('obj_after',obj_motion_dict['trans'][0])


full_motion_dict = {}
# for key_name in tqdm(amass_data.keys()):
for idx in range(1):
    smpl_data_entry = amass_data
    print(smpl_data_entry.keys())
    B = smpl_data_entry['poses'].shape[0]

    start, end = 0, 0

    pose_aa = smpl_data_entry['poses'].copy()[start:][:,:156]
    print(pose_aa.shape, pose_aa.dtype)
    root_trans = smpl_data_entry['trans'].copy()[start:]

    B = pose_aa.shape[0]

    beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy()
    if len(beta.shape) == 2:
        beta = beta[0]
    print(beta,98798969)
    gender = smpl_data_entry.get("gender", "neutral")
    print('gender',gender)
    fps = smpl_data_entry.get("fps", 30.0)

    if isinstance(gender, np.ndarray):
        gender = gender.item()
    print(gender,97897968966969)
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    if gender == "neutral":
        gender_number = [0]
    elif gender == "male":
        gender_number = [1]
    elif gender == "female":
        gender_number = [2]
    else:
        import ipdb
        ipdb.set_trace()
        raise Exception("Gender Not Supported!!")
    gender_number_init = gender_number
    smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
    batch_size = pose_aa.shape[0]
    # pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
    pose_aa = pose_aa
    pose_aa[:,0:3] = (R1 * sRot.from_rotvec(pose_aa[:,0:3])).as_rotvec()
    print('human_before',root_trans[0])
    # root_trans = R1.apply(root_trans)
    print('human_after',root_trans[0])

   
    pose_aa_mj = pose_aa.reshape(-1, 52, 3)[..., smpl_2_mujoco, :].copy()

    num = 1
    if double:
        num = 2
    for idx in range(num):
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 52, 4)

        # gender_number, beta[:], gender = [0], 0, "neutral"
        print("using neutral model")
        beta2 = np.array([
    1.2644597, 0.4629662, -0.9876839, -0.6337372, 1.4846485, -0.05660084,
    1.6636678, -0.7218272, 2.580027, 2.314394, 0.6696473, 0.10814083,
    1.5411701, 0.31156713, -0.6719442, 0.4843772
])
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        smpl_local_robot.write_xml("smplh_humanoid_intercap.xml")
        skeleton_tree = SkeletonTree.from_mjcf("smplh_humanoid_intercap.xml")

        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
        print('human_after_2',root_trans_offset[0])
        root_trans_offset = R1.apply(root_trans_offset.numpy())
        root_trans_offset = torch.from_numpy(root_trans_offset)
        root_trans = R1.apply(root_trans)

        print(root_trans_offset[0],root_trans[0],skeleton_tree.local_translation[0])
        print('huam_object',root_trans_offset[0],obj_motion_dict['trans'][0],root_trans_offset[0]-obj_motion_dict['trans'][0])
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)
        key_name_dump = 'test'
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)  # should fix pose_quat as well here...

            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
            pose_quat = new_sk_state.local_rotation.numpy()

            ############################################################
            
            if idx == 1:
                left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]
                pose_quat_global = pose_quat_global[:, left_to_right_index]
                pose_quat_global[..., 0] *= -1
                pose_quat_global[..., 2] *= -1

                root_trans_offset[..., 1] *= -1
            ############################################################

        new_motion_out = {}
        # pose_quat_global = new_sk_state.global_rotation
        new_motion_out['pose_quat_global'] = pose_quat_global
        new_motion_out['pose_quat'] = pose_quat
        new_motion_out['trans_orig'] = root_trans
        new_motion_out['root_trans_offset'] = root_trans_offset
        # new_motion_out['root_trans_offset'] = root_trans
        new_motion_out['beta'] = beta
        new_motion_out['gender'] = gender
        new_motion_out['pose_aa'] = pose_aa
        new_motion_out['fps'] = fps
        new_motion_out['gender_number'] = gender_number_init
        full_motion_dict[key_name_dump] = new_motion_out


# import ipdb; ipdb.set_trace()
joblib.dump(full_motion_dict, "intercap_test2.pkl")
joblib.dump(obj_motion_dict, "intercap_obj_test2.pkl")