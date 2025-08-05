from ast import Try
from isaacgym import gymapi, gymutil, gymtorch
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
from contact import get_contact
# Adds the current working directory to Python's module search path
# This allows importing modules from the project root directory


from smpl_sim.smpllib.smpl_mujoco_new import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import scipy.ndimage.filters as filters
from typing import List, Optional
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from phc.utils.flags import flags
from easydict import EasyDict
from phc.utils.motion_lib_base import FixHeightMode
from phc.utils.motion_lib_smpl import MotionLibSMPL as MotionLibSMPL
flags.test = True
flags.im_eval = True

robot_cfg = {
    "mesh": False,
    "model": "smplx",
    "rel_joint_lm": False,
    "upright_start": True,
    "remove_toe": False,
    "real_weight_proportion_capsules": True,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}

smpl_local_robot = LocalRobot(
    robot_cfg,
    data_dir="data/smpl",
)
smpl_local_robot2 = LocalRobot(
    robot_cfg,
    data_dir="data/smpl",
)

R1 = np.array([
    [ 1,  0,  0],
    [ 0,  0,  -1],
    [ 0,  1,  0]
])
R1 = sRot.from_matrix(R1)

def convert_obj(obj_data):
    obj_motion_dict = {}
    obj_trans = obj_data['trans']
    pose_aa = obj_data['angles']
    pose_aa = (R1 * sRot.from_rotvec(pose_aa)).as_rotvec()
    pose_quat = sRot.from_rotvec(pose_aa).as_quat()
    obj_motion_dict['rot'] = pose_quat
    obj_motion_dict['trans'] = R1.apply(obj_trans)
    return obj_motion_dict

def convert_human_step1(human_data):
    double = False
    full_motion_dict = {}
    human_data = human_data
    B = human_data['poses'].shape[0]

    start, end = 0, 0

    pose_aa = human_data['poses'].copy()[start:][:,:156]
    root_trans = human_data['trans'].copy()[start:]
    beta = human_data['beta'].copy() if "beta" in human_data else human_data['betas'].copy()
    if len(beta.shape) == 2:
        beta = beta[0]
    gender = human_data.get("gender", "neutral")
    fps = human_data.get("fps", 30.0)

    if isinstance(gender, np.ndarray):
        gender = gender.item()
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
    
    smpl_2_mujoco = [SMPLH_BONE_ORDER_NAMES.index(q) for q in SMPLH_MUJOCO_NAMES if q in SMPLH_BONE_ORDER_NAMES]
    batch_size = pose_aa.shape[0]

    pose_aa[:,0:3] = (R1 * sRot.from_rotvec(pose_aa[:,0:3])).as_rotvec()

    pose_aa_mj = pose_aa.reshape(-1, 52, 3)[..., smpl_2_mujoco, :].copy()

    num = 1
    if double:
        num = 2
    for idx in range(num):
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 52, 4)

        # gender_number, beta[:], gender = [0], 0, "neutral"
        # print("using neutral model")
        # beta2 = np.array([
        # 1.2644597, 0.4629662, -0.9876839, -0.6337372, 1.4846485, -0.05660084,
        # 1.6636678, -0.7218272, 2.580027, 2.314394, 0.6696473, 0.10814083,
        # 1.5411701, 0.31156713, -0.6719442, 0.4843772
        # ])

        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        smpl_local_robot.write_xml("smplh_humanoid_intercap.xml")
        skeleton_tree = SkeletonTree.from_mjcf("smplh_humanoid_intercap.xml")

        root_trans_offset = root_trans + skeleton_tree.local_translation[0].numpy()

        root_trans_offset = torch.from_numpy(R1.apply(root_trans_offset))
        root_trans = R1.apply(root_trans)
        
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
        new_motion_out['gender_number'] = gender_number
        full_motion_dict[key_name_dump] = new_motion_out

    flags.test = True
    flags.im_eval = True
    gender_beta = np.concatenate([gender_number, beta], axis=0)
    device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    
    motion_file = "intercap_test2.pkl"
    joblib.dump(full_motion_dict, motion_file)

    motion_lib_cfg = EasyDict({
                    "motion_file": motion_file,
                    "smpl_type": "smplx",
                    "device": device,
                    "fix_height": FixHeightMode.no_fix,
                    "min_length": -1,
                    "max_length": -1,
                    "im_eval": False,
                    "multi_thread": False ,
                    "randomrize_heading": False,
                })
    motion_lib = MotionLibSMPL(motion_lib_cfg)
    num_motions = 1

    smpl_local_robot2.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
    test_good = f"/tmp/smpl/smplh_humanoid_intercap4.xml"
    smpl_local_robot2.write_xml(test_good)
    skeleton_tree = SkeletonTree.from_mjcf(test_good)
    
    motion_lib.load_motions(skeleton_trees=[skeleton_tree] * num_motions, gender_betas=[torch.from_numpy(gender_beta)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False)
    motion_id = 0
    motion_all = motion_lib.get_motion_state_all(torch.tensor([motion_id]).to(device))
    return motion_all

def convert_final(human_motion_dict, obj_motion_dict, inputdir):
    bsz = human_motion_dict['root_pos'].shape[0]
    data_final = torch.zeros((bsz, 591), dtype=torch.float32)
    data_final[:, :3] = human_motion_dict['root_pos'].squeeze(1)
    data_final[:, 3:7] = human_motion_dict['root_rot'].squeeze(1)
    data_final[:, 9:9+153] = human_motion_dict['dof_pos'].squeeze(1)
    data_final[:, 162:162+52*3] = human_motion_dict['rg_pos'].squeeze(1).reshape(bsz, 52*3)
    data_final[:, 331+52:331+52+52*4] = human_motion_dict['rb_rot'].squeeze(1).reshape(bsz, 52*4)
    
    data_final[:, 318:318+3] = torch.from_numpy(obj_motion_dict['trans'])
    data_final[:, 321:321+4] = torch.from_numpy(obj_motion_dict['rot'])

    human_path = inputdir + '/human.npz'
    obj_path = inputdir + '/object.npz'
    obj_name = inputdir.split('/')[-1].split('_')[1]
    obj_verts = '/'.join(inputdir.split('/')[:-2] + ['objects', obj_name, obj_name + '.obj'])
    human_contact, obj_contact = get_contact(human_path, obj_path, obj_verts)
    data_final[:, 330:331] = obj_contact
    data_final[:, 331:331+52] = human_contact
    return data_final

inputdir = '/data-local/dingbang/phys_hoi_recon/InterAct/data/omomo/sequences_canonical/sub8_smallbox_030'
human_data = np.load(inputdir + '/human.npz', allow_pickle=True)
obj_data = np.load(inputdir + '/object.npz', allow_pickle=True)


obj_motion_dict = convert_obj(obj_data)
human_motion_dict = convert_human_step1(human_data)
data_final = convert_final(human_motion_dict, obj_motion_dict, inputdir)
torch.save(data_final, '/data-local/dingbang/phys_hoi_recon/InterMimic/InterAct/InterCap/sub8_smallbox_031.pt')
