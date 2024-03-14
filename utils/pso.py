import torch
import math
import numpy as np
from PIL import Image

class PSOData():
    def __init__(self, PSOPath):
        self.trans = [None] * 60

        num = 20
        for i in range(num):
            self.trans[i] = np.array(Image.open(PSOPath+"pst_10240_"+str(i)+".tiff"))
            for j in range(6):
                self.trans[i][0][j] = 0.
            self.trans[i] = torch.from_numpy(np.array(self.trans[i])).float().to('cuda')
        for i in range(num, num*2):
            self.trans[i] = np.array(Image.open(PSOPath+"pst_3072_"+str(i-20)+".tiff"))
            for j in range(6):
                self.trans[i][0][j] = 0.
            self.trans[i] = torch.from_numpy(np.array(self.trans[i])).float().to('cuda')
        for i in range(num*2, num*3):
            self.trans[i] = np.array(Image.open(PSOPath+"pst_1024_"+str(i-40)+".tiff"))
            for j in range(6):
                self.trans[i][0][j] = 0.
            self.trans[i] = torch.from_numpy(np.array(self.trans[i])).float().to('cuda')
        self.trans = torch.cat(self.trans, 0)

class PSTSearch():
    def __init__(self):
        self.scaling_coefficient1 = 0.18
        self.scaling_coefficient2 = 0.09
        self.init_fitness = 0.5

        lens = self.scaling_coefficient1 *  self.init_fitness
        self.search_size = [lens] * 6
        self.momentum = 0.9
        self.max_iteration = 20
        self.iteration = 0

        # self.particle_index = [0,1+20,2+40,3,4+20,5+40,6+0,7+20,8+40,
        #                             9+0,10+20,11+40,12+0,13+20,14+40,
        #                             15+0,16+20,17+40,18+0,19+20]
        # This is in camera frame
        self.particle_index = np.arange(40, 60)
        # this is limited by GPU memory
        # yet, we believe there be a way with full batch considering some optimized code
        self.batch = 256
        # These are in gloabl frame
        self.cur_rot = None
        self.cur_trans = None
    
    def update_seach_size(self, T, weight):
        s_tx = abs(T[0]) + 1e-3
        s_ty = abs(T[1]) + 1e-3
        s_tz = abs(T[2]) + 1e-3
        s_qx = abs(T[3]) + 1e-3
        s_qy = abs(T[4]) + 1e-3
        s_qz = abs(T[5]) + 1e-3
        trans_norm = np.sqrt(s_tx*s_tx+s_ty*s_ty+s_tz*s_tz+s_qx*s_qx+s_qy*s_qy+s_qz*s_qz)

        self.search_size[3] = self.scaling_coefficient2*weight*s_qx/trans_norm + 1e-3
        self.search_size[4] = self.scaling_coefficient2*weight*s_qy/trans_norm + 1e-3
        self.search_size[5] = self.scaling_coefficient2*weight*s_qz/trans_norm + 1e-3
        self.search_size[0] = self.scaling_coefficient2*weight*s_tx/trans_norm + 1e-3
        self.search_size[1] = self.scaling_coefficient2*weight*s_ty/trans_norm + 1e-3
        self.search_size[2] = self.scaling_coefficient2*weight*s_tz/trans_norm + 1e-3
    
    def transform_T_cur(self, T):
        # x y z | w x y z
        T[3:] = T[3:] / np.linalg.norm(T[3:])
        q0 = T[3]
        q1 = T[4]
        q2 = T[5]
        q3 = T[6]
        w = self.cur_rot[:, 0]
        x = self.cur_rot[:, 1]
        y = self.cur_rot[:, 2]
        z = self.cur_rot[:, 3]
        self.cur_rot[:, 0] = q0*w - q1*x - q2*y - q3*z
        self.cur_rot[:, 1] = q0*x + q1*w + q2*z - q3*y
        self.cur_rot[:, 2] = q0*y - q1*z + q2*w + q3*x
        self.cur_rot[:, 3] = q0*z + q1*y - q2*x + q3*w
        self.cur_rot = self.cur_rot / torch.norm(self.cur_rot)

        ps_t = torch.zeros((1, 3), device='cuda')
        ps_t[:, 0] = T[0]
        ps_t[:, 1] = T[1]
        ps_t[:, 2] = T[2]
        n = torch.zeros((1, 3, 3), device='cuda')
        n[:, 0, 0] = 1 - 2*q2*q2 - 2*q3*q3
        n[:, 0, 1] = 2*(q1*q2 + q0*q3)
        n[:, 0, 2] = 2*(q1*q3 - q0*q2)
        n[:, 1, 0] = 2*(q1*q2 - q0*q3)
        n[:, 1, 1] = 1 - 2*q1*q1 - 2*q3*q3
        n[:, 1, 2] = 2*(q2*q3 + q0*q1)
        n[:, 2, 0] = 2*(q1*q3 + q0*q2)
        n[:, 2, 1] = 2*(q2*q3 - q0*q1)
        n[:, 2, 2] = 1 - 2*q1*q1 - 2*q2*q2
        self.cur_trans = torch.bmm(n, self.cur_trans.unsqueeze(-1)).squeeze(-1) + ps_t