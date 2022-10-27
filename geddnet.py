# %%
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
CNN for gaze estimation using GeddNet. 
'''




class GEDDnet(nn.Module):

    def __init__(self,
            vgg16_path,
            dropout_p=0.5,
            rf=[[2, 2], [3, 3], [5, 5], [11, 11]],
            num_face=[64, 128, 64, 64, 128, 256, 64],
            r=[[2, 2], [3, 3], [4, 5], [5, 11]],
            num_eye=[64, 128, 64, 64, 128, 256],
            num_comb=[0, 256],
            num_subj=1,
            init_weights=False,
            use_pbias=False):

        super(GEDDnet, self).__init__()

        self.use_pbias = use_pbias
        self.vgg_wei = np.load(vgg16_path, allow_pickle=True)

        self.num_comb = num_comb
        self.num_comb[0] = num_face[-1] + 2*num_eye[-1]
        self.num_face = num_face 
        self.num_eye = num_eye 

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout(p=dropout_p)

        self.face_conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.face_conv1_1.weight.data  = torch.from_numpy(self.vgg_wei['conv1_1_W'].transpose(3, 2, 0, 1))
        self.face_conv1_1.bias.data = torch.from_numpy(self.vgg_wei['conv1_1_b'])

        self.face_conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.face_conv1_2.weight.data = torch.from_numpy(self.vgg_wei['conv1_2_W'].transpose(3, 2, 0, 1))
        self.face_conv1_2.bias.data = torch.from_numpy(self.vgg_wei['conv1_2_b'])

        self.face_conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.face_conv2_1.weight.data = torch.from_numpy(self.vgg_wei['conv2_1_W'].transpose(3, 2, 0, 1))
        self.face_conv2_1.bias.data = torch.from_numpy(self.vgg_wei['conv2_1_b'])

        self.face_conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.face_conv2_2.weight.data = torch.from_numpy(self.vgg_wei['conv2_2_W'].transpose(3, 2, 0, 1))
        self.face_conv2_2.bias.data = torch.from_numpy(self.vgg_wei['conv2_2_b'])

        self.face_conv2_3 = nn.Conv2d(in_channels=num_face[1], out_channels=num_face[2], kernel_size=(1, 1), stride=(1, 1), padding="valid")
        self.face_conv2_3_norm = nn.BatchNorm2d(num_face[2])

        self.face_conv3_1 = nn.Conv2d(in_channels=num_face[2], out_channels=num_face[3], dilation=rf[0], kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.face_conv3_1_norm = nn.BatchNorm2d(num_face[3])

        self.face_conv3_2 = nn.Conv2d(in_channels=num_face[3], out_channels=num_face[3], dilation=rf[1],  kernel_size=(3, 3), stride=(1, 1), padding="same")

        self.face_conv3_2_norm = nn.BatchNorm2d(num_face[3])

        self.face_conv4_1 = nn.Conv2d(in_channels=num_face[3], out_channels=num_face[4], dilation=rf[2], kernel_size=(3, 3),stride=(1, 1), padding="same")

        
        self.face_conv4_1_norm = nn.BatchNorm2d(num_face[4])

        self.face_conv4_2 = nn.Conv2d(in_channels=num_face[4], out_channels=num_face[4], dilation=rf[3], kernel_size=(3, 3),stride=(1, 1), padding="same")

        
        self.face_conv4_2_norm = nn.BatchNorm2d(num_face[4])

        self.face_fc1 = nn.Linear(42*42*num_face[4], num_face[5])

        self.face_fc1_norm = nn.BatchNorm1d(num_face[5])

        self.face_fc2 = nn.Linear(num_face[5], num_face[6])


        self.face_fc2_norm = nn.BatchNorm1d(num_face[6])

        self.eye_conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.eye_conv1_1.weight.data  = torch.from_numpy(self.vgg_wei['conv1_1_W'].transpose(3, 2, 0, 1))
        self.eye_conv1_1.bias.data = torch.from_numpy(self.vgg_wei['conv1_1_b'])
   
        self.eye_conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.eye_conv1_2.weight.data  = torch.from_numpy(self.vgg_wei['conv1_2_W'].transpose(3, 2, 0, 1))
        self.eye_conv1_2.bias.data = torch.from_numpy(self.vgg_wei['conv1_2_b'])

        self.eye_conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.eye_conv2_1.weight.data  = torch.from_numpy(self.vgg_wei['conv2_1_W'].transpose(3, 2, 0, 1))
        self.eye_conv2_1.bias.data = torch.from_numpy(self.vgg_wei['conv2_1_b'])

        self.eye_conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="valid")
        self.eye_conv2_2.weight.data  = torch.from_numpy(self.vgg_wei['conv2_2_W'].transpose(3, 2, 0, 1))
        self.eye_conv2_2.bias.data = torch.from_numpy(self.vgg_wei['conv2_2_b'])


        self.eye_conv2_3 = nn.Conv2d(in_channels=num_eye[1], out_channels=num_eye[2], kernel_size=(1, 1), stride=(1, 1), padding="valid")
        self.eye_conv2_3_norm = nn.BatchNorm2d(num_eye[2])

        self.eye_conv3_1 = nn.Conv2d(in_channels=num_eye[2], out_channels=num_eye[3], dilation=r[0], kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.eye_conv3_1_norm = nn.BatchNorm2d(num_eye[3])
        
        self.eye_conv3_2 = nn.Conv2d(in_channels=num_eye[3], out_channels=num_eye[3], dilation=r[1],  kernel_size=(3, 3), stride=(1, 1), padding="same")
        self.eye_conv3_2_norm = nn.BatchNorm2d(num_eye[3])

        self.eye_conv4_1 = nn.Conv2d(in_channels=num_eye[3], out_channels=num_eye[4], dilation=r[2], kernel_size=(3, 3),stride=(1, 1), padding="same")
        self.eye_conv4_1_norm = nn.BatchNorm2d(num_eye[4])

        self.eye_conv4_2 = nn.Conv2d(in_channels=num_eye[4], out_channels=num_eye[4], dilation=r[3], kernel_size=(2, 2),stride=(1, 1), padding="same")
        self.eye_conv4_2_norm = nn.BatchNorm2d(num_eye[4])

        self.eyel_fc1 = nn.Linear(364*3*num_eye[4], num_eye[5])
        self.eyer_fc1 = nn.Linear(364*3*num_eye[4], num_eye[5])

        self.combined_fc1 = nn.Linear(576, num_comb[1])
        self.combined_fc2 = nn.Linear(num_comb[1], 2)

        self.bias_w_fc =  nn.Parameter(data=torch.rand((num_subj, 2)), requires_grad=True)

        if init_weights:
            nn.init.trunc_normal_(self.face_conv2_3.weight.data, std=0.125)
            nn.init.normal_(self.face_conv2_3.bias.data, std=0.001)
            self.face_conv2_3.bias.data = torch.abs(self.face_conv2_3.bias.data)

            nn.init.trunc_normal_(self.face_conv3_1.weight.data, std=0.06)
            nn.init.normal_(self.face_conv3_1.bias.data, std=0.001)
            self.face_conv3_1.bias.data = torch.abs(self.face_conv3_1.bias.data)

            nn.init.trunc_normal_(self.face_conv3_2.weight.data, std=0.06)
            nn.init.normal_(self.face_conv3_2.bias.data, std=0.001)
            self.face_conv3_2.bias.data = torch.abs(self.face_conv3_2.bias.data)

            nn.init.trunc_normal_(self.face_conv4_1.weight.data, std=0.08)
            nn.init.normal_(self.face_conv4_1.bias.data, std=0.001)
            self.face_conv4_1.bias.data = torch.abs(self.face_conv4_1.bias.data)

            nn.init.trunc_normal_(self.face_conv4_2.weight.data, std=0.07)
            nn.init.normal_(self.face_conv4_2.bias.data, std=0.001)
            self.face_conv4_2.bias.data = torch.abs(self.face_conv4_2.bias.data)

            nn.init.trunc_normal_(self.face_fc1.weight.data, std=0.035)
            nn.init.normal_(self.face_fc1.bias.data, std=0.001)
            self.face_fc1.bias.data = torch.abs(self.face_fc1.bias.data)

            nn.init.trunc_normal_(self.face_fc1.weight.data, std=0.1)
            nn.init.normal_(self.face_fc1.bias.data, std=0.001)
            self.face_fc1.bias.data = torch.abs(self.face_fc1.bias.data)

            nn.init.trunc_normal_(self.eye_conv2_3.weight.data, std=0.125)
            nn.init.normal_(self.eye_conv2_3.bias.data, std=0.001)
            self.eye_conv2_3.bias.data = torch.abs(self.eye_conv2_3.bias.data)

            nn.init.trunc_normal_(self.eye_conv3_1.weight.data, std=0.06)
            nn.init.normal_(self.eye_conv3_1.bias.data, std=0.001)
            self.eye_conv3_1.bias.data = torch.abs(self.eye_conv3_1.bias.data)

            nn.init.trunc_normal_(self.eye_conv3_2.weight.data, std=0.06)
            nn.init.normal_(self.eye_conv3_2.bias.data, std=0.001)
            self.eye_conv3_2.bias.data = torch.abs(self.eye_conv3_2.bias.data)

            nn.init.trunc_normal_(self.eye_conv4_1.weight.data, std=0.06)
            nn.init.normal_(self.eye_conv4_1.bias.data, std=0.001)
            self.eye_conv4_1.bias.data = torch.abs(self.eye_conv4_1.bias.data)

            nn.init.trunc_normal_(self.eye_conv4_2.weight.data, std=0.04)
            nn.init.normal_(self.eye_conv4_2.bias.data, std=0.001)
            self.eye_conv4_2.bias.data = torch.abs(self.eye_conv4_2.bias.data)

            nn.init.trunc_normal_(self.eyel_fc1.weight.data, std=0.026)
            nn.init.normal_(self.eyel_fc1.bias.data, std=0.001)
            self.eyel_fc1.bias.data = torch.abs(self.eyel_fc1.bias.data)

            nn.init.trunc_normal_(self.eyer_fc1.weight.data, std=0.026)
            nn.init.normal_(self.eyer_fc1.bias.data, std=0.001)
            self.eyer_fc1.bias.data = torch.abs(self.eyer_fc1.bias.data)

            nn.init.trunc_normal_(self.combined_fc1.weight.data, std=0.07)
            nn.init.normal_(self.combined_fc1.bias.data, std=0.001)
            self.combined_fc1.bias.data = torch.abs(self.combined_fc1.bias.data)

            nn.init.trunc_normal_(self.combined_fc2.weight.data, std=0.125)
            nn.init.normal_(self.combined_fc2.bias.data, std=0.001)
            self.combined_fc2.bias.data = torch.abs(self.combined_fc2.bias.data)

            nn.init.trunc_normal_(self.bias_w_fc.data, std=0.125)

        self.float()

    def forward(self, X_face, X_leye, X_reye, sunj_id):
        X_face = F.relu(self.face_conv1_1(X_face))
        X_face = F.relu(self.face_conv1_2(X_face))
        X_face = self.max_pool(X_face)

        X_face = F.relu(self.face_conv2_1(X_face))
        X_face = F.relu(self.face_conv2_2(X_face)) / 100
        X_face = F.relu(self.face_conv2_3(X_face))
        X_face = self.face_conv2_3_norm(X_face)
        
        X_face = F.relu(self.face_conv3_1(X_face))
        X_face = self.face_conv3_1_norm(X_face)

        X_face = F.relu(self.face_conv3_2(X_face))
        X_face = self.face_conv3_2_norm(X_face)
        X_face = F.relu(self.face_conv4_1(X_face))
        X_face = self.face_conv4_1_norm(X_face)
        X_face = F.relu(self.face_conv4_2(X_face))
        X_face = self.face_conv4_2_norm(X_face)

        X_face = F.relu(self.face_fc1(X_face.reshape(-1, 42*42*self.num_face[4])))
        X_face = self.dropout(X_face)
        X_face = F.relu(self.face_fc2(X_face))

        X_leye = F.relu(self.eye_conv1_1(X_leye))
        X_leye = F.relu(self.eye_conv1_2(X_leye))
        X_leye = self.max_pool(X_leye)

        X_leye = F.relu(self.eye_conv2_1(X_leye))
        X_leye = F.relu(self.eye_conv2_2(X_leye)) / 100

        X_leye = F.relu(self.eye_conv2_3(X_leye))
        X_leye = self.eye_conv2_3_norm(X_leye)

        X_leye = F.relu(self.eye_conv3_1(X_leye))
        X_leye = self.eye_conv3_1_norm(X_leye)

        X_leye = F.relu(self.eye_conv3_2(X_leye))
        X_leye = self.eye_conv3_2_norm(X_leye)

        X_leye = F.relu(self.eye_conv4_1(X_leye))
        X_leye = self.eye_conv4_1_norm(X_leye)

        X_leye = F.relu(self.eye_conv4_2(X_leye))
        X_leye = self.eye_conv4_2_norm(X_leye)

        X_leye = F.relu(self.eyel_fc1(X_leye.reshape(-1, 364*3*self.num_eye[4])))

        X_reye = F.relu(self.eye_conv1_1(X_reye))
        X_reye = F.relu(self.eye_conv1_2(X_reye))
        X_reye = self.max_pool(X_reye)

        X_reye = F.relu(self.eye_conv2_1(X_reye))
        X_reye = F.relu(self.eye_conv2_2(X_reye)) / 100

        X_reye = F.relu(self.eye_conv2_3(X_reye))
        X_reye = self.eye_conv2_3_norm(X_reye)

        X_reye = F.relu(self.eye_conv3_1(X_reye))
        X_reye = self.eye_conv3_1_norm(X_reye)

        X_reye = F.relu(self.eye_conv3_2(X_reye))
        X_reye = self.eye_conv3_2_norm(X_reye)

        X_reye = F.relu(self.eye_conv4_1(X_reye))
        X_reye = self.eye_conv4_1_norm(X_reye)

        X_reye = F.relu(self.eye_conv4_2(X_reye))
        X_reye = self.eye_conv4_2_norm(X_reye)

        X_reye = F.relu(self.eyer_fc1(X_reye.reshape(-1, 364*3*self.num_eye[4])))

        X_combined = torch.cat((X_face, X_leye, X_reye), 1)

        X_combined = self.dropout(X_combined)
        X_combined = F.relu(self.combined_fc1(X_combined.reshape(-1, 576)))
        X_combined = self.dropout(X_combined)

        t_hat = F.relu(self.combined_fc2(X_combined))
        g_hat = t_hat
        if self.use_pbias:
            b_hat = torch.matmul(sunj_id, self.bias_w_fc)
            g_hat +=  b_hat

        return g_hat, t_hat
