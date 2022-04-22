# %%
import torch 
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np




vgg_wei = np.load("vgg16_weights.npz", allow_pickle=True)




class GEDDnet(nn.Module):

    def __init__(self,
            rf=[[2, 2], [3, 3], [5, 5], [11, 11]],
            num_face=[64, 128, 64, 64, 128, 256, 64],
            r=[[2, 2], [3, 3], [4, 5], [5, 11]],
            num_eye=[64, 128, 64, 64, 128, 256],
            num_comb=[0, 256]):

        super(GEDDnet, self).__init__()


        self.num_comb = [0, 256]
        self.num_comb[0] = num_face[-1] + 2*num_eye[-1]

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.face_conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.face_conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))

        self.face_conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1))
        self.face_conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1))

        self.face_conv2_3 = nn.Conv2d(in_channels=num_face[1], out_channels=num_face[2], kernel_size=(1, 1), stride=(1, 1))
        
        self.face_conv3_1 = nn.Conv2d(in_channels=num_face[2], out_channels=num_face[3], dilation=rf[0], kernel_size=(3, 3), stride=(1, 1))
        self.face_conv3_2 = nn.Conv2d(in_channels=num_face[3], out_channels=num_face[3], dilation=rf[1],  kernel_size=(3, 3), stride=(1, 1))

        self.face_conv4_1 = nn.Conv2d(in_channels=num_face[3], out_channels=num_face[4], dilation=rf[2], kernel_size=(3, 3),stride=(1, 1))
        self.face_conv4_2 = nn.Conv2d(in_channels=num_face[4], out_channels=num_face[4], dilation=rf[3], kernel_size=(3, 3),stride=(1, 1))

        self.face_fc1 = nn.Linear(12*12*num_face[4], num_face[5])
        self.face_fc2 = nn.Linear(num_face[5], num_face[6])

        self.leye_conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.leye_conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))

        self.leye_conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1))
        self.leye_conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1))

        self.leye_conv2_3 = nn.Conv2d(in_channels=num_eye[1], out_channels=num_eye[2], kernel_size=(1, 1), stride=(1, 1))
        
        self.leye_conv3_1 = nn.Conv2d(in_channels=num_eye[2], out_channels=num_eye[3], dilation=r[0], kernel_size=(3, 3), stride=(1, 1))
        self.leye_conv3_2 = nn.Conv2d(in_channels=num_eye[3], out_channels=num_eye[3], dilation=r[1],  kernel_size=(3, 3), stride=(1, 1))

        self.leye_conv4_1 = nn.Conv2d(in_channels=num_eye[3], out_channels=num_eye[4], dilation=r[2], kernel_size=(3, 3),stride=(1, 1))
        self.leye_conv4_2 = nn.Conv2d(in_channels=num_eye[4], out_channels=num_eye[4], dilation=r[3], kernel_size=(2, 2),stride=(1, 1))

        self.leye_fc1 = nn.Linear(31*3*num_eye[4], num_eye[5])

        self.reye_conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.reye_conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))

        self.reye_conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1))
        self.reye_conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1))

        self.reye_conv2_3 = nn.Conv2d(in_channels=num_eye[1], out_channels=num_eye[2], kernel_size=(1, 1), stride=(1, 1))
        
        self.reye_conv3_1 = nn.Conv2d(in_channels=num_eye[2], out_channels=num_eye[3], dilation=r[0], kernel_size=(3, 3), stride=(1, 1))
        self.reye_conv3_2 = nn.Conv2d(in_channels=num_eye[3], out_channels=num_eye[3], dilation=r[1],  kernel_size=(3, 3), stride=(1, 1))

        self.reye_conv4_1 = nn.Conv2d(in_channels=num_eye[3], out_channels=num_eye[4], dilation=r[2], kernel_size=(3, 3),stride=(1, 1))
        self.reye_conv4_2 = nn.Conv2d(in_channels=num_eye[4], out_channels=num_eye[4], dilation=r[3], kernel_size=(2, 2),stride=(1, 1))

        self.reye_fc1 = nn.Linear(31*3*num_eye[4], num_eye[5])

        self.combined_fc1 = nn.Linear(576, num_comb[1])
        self.combined_fc2 = nn.Linear(num_comb[1], 2)

    def forward(self, X_face, X_leye, X_reye):
        X_face = F.relu(self.face_conv1_1(X_face))
        X_face = F.relu(self.face_conv1_2(X_face))
        X_face = self.max_pool(X_face)

        X_face = F.relu(self.face_conv2_1(X_face))
        X_face = F.relu(self.face_conv2_2(X_face))
        X_face = F.relu(self.face_conv2_3(X_face))
        X_face = F.relu(self.face_conv3_1(X_face))
        X_face = F.relu(self.face_conv3_2(X_face))
        X_face = F.relu(self.face_conv4_1(X_face))
        X_face = F.relu(self.face_conv4_2(X_face))

        X_face = F.relu(self.face_fc1(X_face.view(-1)))
        X_face = F.relu(self.face_fc2(X_face))

        X_leye = F.relu(self.leye_conv1_1(X_leye))
        X_leye = F.relu(self.leye_conv1_2(X_leye))
        X_leye = self.max_pool(X_leye)

        X_leye = F.relu(self.leye_conv2_1(X_leye))
        X_leye = F.relu(self.leye_conv2_2(X_leye))
        X_leye = F.relu(self.leye_conv2_3(X_leye))
        X_leye = F.relu(self.leye_conv3_1(X_leye))
        X_leye = F.relu(self.leye_conv3_2(X_leye))
        X_leye = F.relu(self.leye_conv4_1(X_leye))
        X_leye = F.relu(self.leye_conv4_2(X_leye))

        X_leye = F.relu(self.leye_fc1(X_leye.view(-1)))

        X_reye = F.relu(self.reye_conv1_1(X_reye))
        X_reye = F.relu(self.reye_conv1_2(X_reye))
        X_reye = self.max_pool(X_reye)

        X_reye = F.relu(self.reye_conv2_1(X_reye))
        X_reye = F.relu(self.reye_conv2_2(X_reye))
        X_reye = F.relu(self.reye_conv2_3(X_reye))
        X_reye = F.relu(self.reye_conv3_1(X_reye))
        X_reye = F.relu(self.reye_conv3_2(X_reye))
        X_reye = F.relu(self.reye_conv4_1(X_reye))
        X_reye = F.relu(self.reye_conv4_2(X_reye))

        X_reye = F.relu(self.reye_fc1(X_reye.view(-1)))

        X_combined = torch.cat((X_face, X_leye, X_reye), 0)
        X_combined = F.relu(self.combined_fc1(X_combined))
        X_combined = F.relu(self.combined_fc2(X_combined))

        return X_combined



