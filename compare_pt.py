from logging import root
import torch 
import joblib
from scipy.spatial.transform import Rotation as sRot
data1 = torch.load('/data-local/dingbang/phys_hoi_recon/InterMimic/InterAct/InterCap/sub8_smallbox_030.pt')
data2 = torch.load('/data-local/dingbang/phys_hoi_recon/InterMimic/InterAct/OMOMO_new/sub8_smallbox_030.pt')
data3 = torch.load('/data-local/dingbang/phys_hoi_recon/InterMimic/InterAct/InterCap/sub8_smallbox_031.pt')
print((data3-data1).abs().sum())

# data1 = torch.load('/data-local/dingbang/InterMimic/InterAct/OMOMO_new/sub2_largetable_018.pt')[:,:]
# data2 = torch.load('/data-local/dingbang/phys_hoi_recon/InterMimic/OMOMO_new/sub2_largetable_018.pt')[:,:]

offset = data1[:,:3][:1] - data2[:,:3][:1]
print('global_offset;', offset)

print('root_trans',data1[0,:3],data2[0,:3]+offset)

print('root_rot',data1[:,3:7][0], data2[:,3:7][0])

print('dof_pos',data1[:,9:9+153][0]- data2[:,9:9+153][0])

print('dof_pos',data1[:,9:9+3][0], data2[:,9:9+3][0])

print('body_pos',data1[:, 162:162+52*3][10], data2[:, 162:162+52*3][10] + offset.expand(52,-1).reshape(52*3))

print('body_pos_diff',data1[:, 162:162+52*3][10]- data2[:, 162:162+52*3][10] - offset.expand(52,-1).reshape(52*3))

print('body_rot',data1[:, 331+52:331+52+52*4][0]- data2[:, 331+52:331+52+52*4][0])

print(data2[:,3:7][0],data2[:,331+52:331+52+4][0])
root_rot = data2[:,3:7][0].detach().numpy()
root_rot = sRot.from_quat(root_rot)

print((sRot.from_euler('XYZ',data2[:,9:9+3][0].detach().numpy(), degrees=True)).as_quat())

print(data2[:,331+52+4:331+52+8][0].detach().numpy())




# print(data1.shape, data2.shape)
# print((data1-data2).abs().sum(),7777777)
# error = (data1-data2).abs()>0.01
# print(torch.where(error)[0])  # Print indices where error is True
# print(data1[0:3],data2[0:3])
# print(data1[216:256],data2[216:256])

# data3 = joblib.load('/data-local/dingbang/phys_hoi_recon/PHC/intercap_test2.pkl')
# data4 = torch.load('/data-local/dingbang/phys_hoi_recon/InterMimic/InterAct/InterCap/sub7_smallbox_002.pt')
# root_pos = data1[:,:3][0] - data2[:,:3][0]

# print(data1[:,:3][10], data2[:,:3][10],data4[:,:3][10])
# print(data3['test']['trans_orig'][10],data3['test']['root_trans_offset'][10])
# print(data1[:,162:165][10], data2[:,162:165][10])


# print(data1[:,318:321][10], data2[:,318:321][10],data4[:,318:321][10])
# print(data1[:,:3][10]-data1[:,318:321][10], data2[:,:3][10]-data2[:,318:321][10])
# print(data1[:,321:325][10], data2[:,321:325][10],8765)


# print(89662865296492)
# print(data1[:, 162:162+52*3][0]- data2[:, 162:162+52*3][0])
# print(data1[:,:3][0],data2[:,:3][0])
# print(1111)
# print
# root_rot1 = data1[:,3:7][20].detach().numpy()
# root_rot1 = sRot.from_quat(root_rot1).as_rotvec()
# root_rot2 = data2[:,3:7][20].detach().numpy()
# root_rot2 = sRot.from_quat(root_rot2).as_rotvec()
# root_rot3 = data3['test']['pose_aa'][20][:3]
# print(root_rot1, root_rot2, root_rot3)

# root_dof = data2[:,9:9+153][0]
# # print(root_dof)


# body_rot = data2[:, 331+52:331+52+52*4][10]- data1[:, 331+52:331+52+52*4][10]
# print(22222)
# print(body_rot)
# print((data2[:, 162:162+52*3][50]- data1[:, 162:162+52*3][50]).reshape(52, 3)-(data2[:, 162:162+52*3][0]- data1[:, 162:162+52*3][0]).reshape(52, 3)[0])
# # # print(root_pos.shape, root_pos.mean(), root_pos.std())

# # contact_obj = data1[:,330:331] - data2[:,330:331]
# # print(contact_obj)
# # contact_obj_all = torch.sum(data2[:,331:331+52][50]==0.0)
# # print(contact_obj_all)


# print(data1[:, 9:9+52*3][20]- data2[:, 9:9+52*3][20])