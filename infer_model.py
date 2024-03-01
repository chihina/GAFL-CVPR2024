from backbone.backbone import *
from backbone.backbone_kinetics import My3DResNet18
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
import collections
import sys
import os
import cv2
import torchvision
import torchvision.models as models
from torchvision.utils import save_image

from unipose.model.unipose import unipose
from unipose.utils.utils import get_kpts

from infer_module.higcin_infer_module import CrossInferBlock
from infer_module.dynamic_infer_module import Dynamic_Person_Inference, Hierarchical_Dynamic_Inference, Multi_Dynamic_Inference

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class GroupRelation_volleyball(nn.Module):
    """
    main module of GR learning for the volleyball dataset
    """

    def __init__(self, cfg):
        super(GroupRelation_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        self.backbone_type = self.cfg.backbone

        self.backbone_type = self.cfg.backbone
        backbone_pretrain = True
        # backbone_pretrain = False
        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=backbone_pretrain)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=backbone_pretrain)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=backbone_pretrain)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=backbone_pretrain)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=backbone_pretrain)
        elif cfg.backbone == '3d_res18_kinetics':
            self.backbone = My3DResNet18(pretrained=backbone_pretrain)
        elif cfg.backbone == 'c3d_ucf101':
            from backbone.backbone_ucf101 import C3D
            self.backbone = C3D(pretrained=backbone_pretrain)
        else:
            assert False

        if not cfg.train_backbone:
            if cfg.backbone == 'vgg16_ucf101':
                for p in self.backbone.collect_params().values():
                    p.grad_req = 'null'
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)

        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.use_ind_feat_crop = self.cfg.use_ind_feat_crop

        if self.use_ind_feat_crop == 'roi_multi':
            self.ind_ext_feat_dim = K * K * D
        elif self.use_ind_feat_crop == 'crop_single':
            self.ind_ext_feat_dim = NFB

        # addtional parameters
        self.eval_only = self.cfg.eval_only
        if self.eval_only:
            self.eval_mask_num = self.cfg.eval_mask_num
        self.use_random_mask = self.cfg.use_random_mask
        if self.use_random_mask:
            self.random_mask_type = self.cfg.random_mask_type
        self.use_ind_feat = self.cfg.use_ind_feat
        self.use_trans = self.cfg.use_trans
        self.use_same_enc_dual_path = self.cfg.use_same_enc_dual_path
        self.trans_head_num = self.cfg.trans_head_num
        self.trans_layer_num = self.cfg.trans_layer_num
        self.people_pool_type = self.cfg.people_pool_type
        self.use_pos_cond = self.cfg.use_pos_cond
        self.use_tmp_cond = self.cfg.use_tmp_cond
        self.final_head_mid_num = self.cfg.final_head_mid_num
        self.use_recon_loss = self.cfg.use_recon_loss
        self.use_recon_diff_loss = self.cfg.use_recon_diff_loss
        self.use_act_loss = self.cfg.use_act_loss
        self.use_pose_loss = self.cfg.use_pose_loss
        self.use_jae_loss = self.cfg.use_jae_loss
        self.use_old_act_rec = self.cfg.old_act_rec
        self.use_res_connect = self.cfg.use_res_connect
        self.use_gen_iar = self.cfg.use_gen_iar
        if self.use_gen_iar:
            self.gen_iar_ratio = self.cfg.gen_iar_ratio

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        if self.use_trans:
            self.temporal_transformer_encoder = nn.ModuleList([
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=self.trans_head_num, batch_first=True, dropout=0.0), num_layers=self.trans_layer_num),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=self.trans_head_num, batch_first=True, dropout=0.0), num_layers=self.trans_layer_num),
            ])
            self.spatial_transformer_encoder = nn.ModuleList([
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=self.trans_head_num, batch_first=True, dropout=0.0), num_layers=self.trans_layer_num),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=self.trans_head_num, batch_first=True, dropout=0.0), num_layers=self.trans_layer_num)
            ])
        
        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])
        self.spatial_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        # IAR with locations
        self.pos_enc_iar = positionalencoding2d(2*NFB, H, W)
        self.tem_enc_iar = positionalencoding1d(2*NFB, 100)

        # keypoint estimation
        if self.use_pose_loss:
            pose_model_type = 'LSP'
            # pose_model_type = 'MPII'
            if pose_model_type == 'LSP':
                self.cfg.keypoint_num = 14
            elif pose_model_type == 'MPII':
                self.cfg.keypoint_num = 16

            self.pose_model = unipose(pose_model_type, num_classes=self.cfg.keypoint_num, backbone='resnet', output_stride=16, 
                            sync_bn=True, freeze_bn=False, stride=8)
            checkpoint = torch.load(os.path.join('pretrained_weights', f'UniPose_{pose_model_type}.tar'))
            p = checkpoint['state_dict']
            state_dict = self.pose_model.state_dict()
            pose_model_dict = {}
            for k,v in p.items():
                if k in state_dict:
                    pose_model_dict[k] = v
            state_dict.update(pose_model_dict)
            self.pose_model.load_state_dict(state_dict)
            self.pose_model = self.pose_model.to(device=torch.device('cuda'))
            self.pose_model.eval()

        # recog_head_mid_num
        if self.final_head_mid_num == 0:
            self.fc_actions_with_gr = nn.Sequential(
                nn.Linear(NFB*2, self.cfg.num_actions),
            )
            if self.use_gen_iar:
                self.fc_actions_gen_iar = nn.Sequential(
                    nn.Linear(NFB, self.cfg.num_actions),
                )
            if self.use_recon_loss or self.use_recon_diff_loss:
                self.fc_recon_with_gr = nn.Sequential(
                    nn.Linear(NFB*2, self.ind_ext_feat_dim),
                )
            if self.use_pose_loss:
                self.fc_pose_with_gr = nn.Sequential(
                    nn.Linear(NFB*2, self.cfg.keypoint_num*2),
                )
        else:
            if self.use_old_act_rec:
                self.fc_actions_with_gr = nn.Sequential(
                        nn.Linear(NFB*2, NFB),
                        nn.ReLU(),
                        nn.Linear(NFB, NFB),
                        nn.ReLU(),
                        nn.Linear(NFB, self.cfg.num_actions),
                )
                if self.use_recon_loss:
                    self.fc_recon_with_gr = nn.Sequential(
                        nn.Linear(NFB*2, NFB),
                        nn.ReLU(),
                        nn.Linear(NFB, NFB),
                        nn.ReLU(),
                        nn.Linear(NFB, self.ind_ext_feat_dim),
                    )
            else:
                self.final_head_mid = nn.Sequential()
                for i in range(self.final_head_mid_num):
                    if i == 0:
                        self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB*2, NFB))
                    else:
                        self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB, NFB))
                    self.final_head_mid.add_module('relu%d' % i, nn.ReLU())

                self.fc_actions_with_gr = nn.Sequential(
                        self.final_head_mid,
                        nn.Linear(NFB, self.cfg.num_actions),
                )

                if self.use_gen_iar:
                    self.final_head_mid_gen = nn.Sequential()
                    for i in range(self.final_head_mid_num):
                        self.final_head_mid_gen.add_module('fc%d' % i, nn.Linear(NFB, NFB))
                        self.final_head_mid_gen.add_module('relu%d' % i, nn.ReLU())

                    self.fc_actions_gen_iar = nn.Sequential(
                            self.final_head_mid_gen,
                            nn.Linear(NFB, self.cfg.num_actions),
                    )

                if self.use_recon_loss or self.use_recon_diff_loss:
                    self.fc_recon_with_gr = nn.Sequential(
                            self.final_head_mid,
                            nn.Linear(NFB, self.ind_ext_feat_dim),
                    )
                
                if self.use_pose_loss:
                    self.fc_pose_with_gr = nn.Sequential(
                            self.final_head_mid,
                            nn.Linear(NFB, self.cfg.keypoint_num*2),
                    )
        
        # GAR with group relation
        self.training_stage = self.cfg.training_stage
        if self.training_stage == 3:
            self.fc_activities = nn.Sequential(
                nn.Linear(NFB*2, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, cfg.num_activities),
            )
            self.fc_actions = nn.Sequential(
                    nn.Linear(NFB*2, cfg.num_actions),
            )
            if self.use_jae_loss:
                self.fc_jae = nn.Sequential(
                    nn.Linear(NFB*2, NFB),
                    nn.ReLU(),
                    nn.Linear(NFB, NFB),
                    nn.ReLU(),
                    nn.Linear(NFB, 2),
                )

        # explainablity optimization
        self.use_exaplainable_mask = False

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
    
    def set_explainable_type(self, mask_act_type):
        self.use_exaplainable_mask = True
        self.mask_act_type = mask_act_type

    def update_explainable_mask(self, xai_person_mask):
        # w/o norm 0 to 1
        self.xai_person_mask = xai_person_mask

    def forward(self, batch_data):
        # images_in, boxes_in, images_person_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        images_person_in = batch_data['images_person_in']

        if self.use_exaplainable_mask:
            if 'perturbation' in self.mask_act_type:
                xai_person_mask = batch_data['xai_person_mask']
            elif 'backprop' in self.mask_act_type:
                xai_person_mask = self.xai_person_mask

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        PH, PW = self.cfg.person_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        if self.use_ind_feat_crop in ['roi_multi']:
            # Use backbone to extract features of images_in
            # Pre-precess first
            images_in_flat = prep_images(images_in_flat)
            outputs = self.backbone(images_in_flat)

            # Build  features
            features_multiscale = []
            for features in outputs:
                if features.shape[2:4] != torch.Size([OH, OW]):
                    features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
                features_multiscale.append(features)
            features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

            # RoI Align
            boxes_in_flat.requires_grad = False
            boxes_idx_flat.requires_grad = False
            boxes_features = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*N, D, K, K,
            boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

            # Embedding
            boxes_features_emb = self.fc_emb_1(boxes_features)  # B,T,N, NFB
            boxes_features_emb = self.nl_emb_1(boxes_features_emb)
            boxes_features_emb = F.relu(boxes_features_emb)

        # for 3d resnet
        elif self.use_ind_feat_crop in ['crop_single']:
            if self.backbone_type == 'c3d_ucf101':
                # (B, T, N, 3, H, W)
                images_person_in_flat = prep_images_c3d_ucf101(images_person_in)
                _, _, _, _, RPH, RPW = images_person_in_flat.shape
                images_person_in_flat = images_person_in_flat.permute(0, 2, 3, 1, 4, 5)
                images_person_in_flat = images_person_in_flat.reshape(B*N, 3, T, RPH, RPW)
                outputs = self.backbone(images_person_in_flat)
                outputs = outputs.reshape(B, N, 1, -1)
                outputs = outputs.permute(0, 2, 1, 3)
                boxes_features = outputs.repeat(1, T, 1, 2)
                boxes_features_emb = boxes_features
            else:
                # (B, T, 3, N, 3, H, W)
                images_person_in_pad = torch.zeros(B, T, 3, N, 3, PH, PW).to(device=images_person_in.device)
                for t_idx in range(1, T-1):
                    images_person_in_cut = images_person_in[:, t_idx-1:t_idx+2]
                    images_person_in_pad[:, t_idx, :, :, :, :, :] = images_person_in_cut
                images_person_in_pad[:, 0, :, :, :, :, :] = images_person_in[:, 0:3]
                images_person_in_pad[:, T-1, :, :, :, :, :] = images_person_in[:, T-3:T]
                images_person_in_flat = prep_images_3dresnet(images_person_in_pad)
                _, _, _, _, _, RPH, RPW = images_person_in_flat.shape
                images_person_in_flat = images_person_in_flat.permute(0, 1, 3, 4, 2, 5, 6)
                images_person_in_flat = images_person_in_flat.reshape(B*T*N, 3, 3, RPH, RPW)
                outputs = self.backbone(images_person_in_flat)
                outputs = outputs.reshape(B, T, N, -1)
                boxes_features = outputs.repeat(1, 1, 1, 2)
                boxes_features_emb = boxes_features
        else:
            assert False, 'use_ind_feat_crop error'

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        if self.use_ind_feat == 'loc_and_app':
            ind_feat_set = boxes_features_emb + ind_loc_feat + ind_tem_feat
        elif self.use_ind_feat == 'loc':
            ind_feat_set = ind_loc_feat + ind_tem_feat
        elif self.use_ind_feat == 'app':
            ind_feat_set = boxes_features_emb + ind_tem_feat
        
        # generate padding masks for transformer
        people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()

        # generate explainable mask
        if self.use_exaplainable_mask and 'original' in self.mask_act_type:
            if 'backprop' in self.mask_act_type:
                # ind_feat_set = ind_feat_set * xai_person_mask.view(B, 1, N, 1).expand(B, T, N, NFB)
                xai_person_mask_sigmoid = torch.sigmoid(xai_person_mask)
                ind_feat_set = ind_feat_set * xai_person_mask_sigmoid.view(B, 1, N, 1).expand(B, T, N, NFB)
            if 'perturbation' in self.mask_act_type:
                xai_person_mask_expand = xai_person_mask.view(B, 1, N).expand(B, T, N)
                people_pad_mask = torch.logical_or(people_pad_mask.bool(), (~xai_person_mask_expand.bool())).bool()

        ind_feat_set_upper = ind_feat_set.view(B*T, N, NFB)
        ind_feat_set_bottom = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)

        # generate random mask for gr learning
        self.adapt_mask = (self.use_random_mask) and (not self.eval_only) and (self.training_stage == 2)

        # mask during training
        if self.adapt_mask:
            if 'active' in self.random_mask_type:
                actions_in = batch_data['actions_in'].reshape(B, T, N)
                if self.random_mask_type == 'active':
                    random_mask = actions_in!=7
                elif self.random_mask_type == 'active_inv':
                    random_mask = actions_in==7
                else:
                    random_max_num = int(self.random_mask_type.split('_')[-1])
                    random_mask = actions_in==random_max_num
            else:
                random_mask = torch.zeros(B, N).to(device=people_pad_mask.device)
                random_max_num = int(self.random_mask_type.split('_')[-1])
                for b in range(B):
                    people_pad_mask_b = people_pad_mask[b, 0]
                    people_pad_mask_b_idx = torch.where(people_pad_mask_b==0)[0]
                    people_pad_mask_b_idx_shuffle = people_pad_mask_b_idx[torch.randperm(people_pad_mask_b_idx.shape[0])]
                    random_max_num_update = int((random_max_num/N)*torch.sum(people_pad_mask_b==0))
                    mask_people_num = people_pad_mask_b_idx_shuffle[:random_max_num_update]
                    random_mask[b, mask_people_num] = True
                random_mask = random_mask.view(B, 1, N).expand(B, T, N)

            people_pad_mask_jud = torch.logical_or(people_pad_mask.bool(), random_mask.bool()).bool()
            all_mask_flag = torch.sum(torch.sum(people_pad_mask_jud==0, dim=(-1))==0)>0
            if not all_mask_flag:
                people_pad_mask = torch.logical_or(people_pad_mask.bool(), random_mask.bool()).bool()

        # mask during inference
        if self.eval_only:
            random_mask = torch.zeros(B, N).to(device=people_pad_mask.device)
            random_max_num = self.eval_mask_num
            for b in range(B):
                people_pad_mask_b = people_pad_mask[b, 0]
                people_pad_mask_b_idx = torch.where(people_pad_mask_b==0)[0]
                people_pad_mask_b_idx_shuffle = people_pad_mask_b_idx[torch.randperm(people_pad_mask_b_idx.shape[0])]
                random_max_num_update = int((random_max_num/N)*torch.sum(people_pad_mask_b==0))
                mask_people_num = people_pad_mask_b_idx_shuffle[:random_max_num_update]
                random_mask[b, mask_people_num] = True
            random_mask = random_mask.view(B, 1, N).expand(B, T, N)

            people_pad_mask_jud = torch.logical_or(people_pad_mask.bool(), random_mask.bool()).bool()
            all_mask_flag = torch.sum(torch.sum(people_pad_mask_jud==0, dim=(-1))==0)>0
            if not all_mask_flag:
                people_pad_mask = torch.logical_or(people_pad_mask.bool(), random_mask.bool()).bool()

        people_pad_mask_spatial = people_pad_mask.view(B*T, N)
        people_pad_mask_temporal = torch.transpose(people_pad_mask, 1, 2).contiguous().view(B*N, T)

        if self.use_trans:
            # upper branch
            ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
            ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_enc_upper)
            ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
            ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
            ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
            ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
            ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
            ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

            # bottom branch
            if self.use_same_enc_dual_path:
                ind_feat_enc_bottom = self.temporal_transformer_encoder[0](ind_feat_set_bottom)
            else:
                ind_feat_enc_bottom = self.temporal_transformer_encoder[1](ind_feat_set_bottom)
            ind_feat_enc_bottom_res = self.temporal_transformer_mlp[0](ind_feat_enc_bottom)
            ind_feat_enc_bottom = ind_feat_set_bottom + ind_feat_enc_bottom_res
            ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, N, T, NFB)                
            ind_feat_enc_bottom = torch.transpose(ind_feat_enc_bottom, 1, 2).contiguous().view(B*T, N, NFB)
            if self.use_same_enc_dual_path:
                ind_feat_enc_bottom = self.spatial_transformer_encoder[0](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)
            else:
                ind_feat_enc_bottom = self.spatial_transformer_encoder[1](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)

            ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, T, N, NFB)
        else:
            # upper branch
            # ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
            ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_set_upper)
            ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
            ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
            ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
            # ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
            ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
            ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

            # bottom branch
            # ind_feat_enc_bottom = self.temporal_transformer_encoder[1](ind_feat_set_bottom)
            ind_feat_enc_bottom_res = self.temporal_transformer_mlp[0](ind_feat_set_bottom)
            ind_feat_enc_bottom = ind_feat_set_bottom + ind_feat_enc_bottom_res
            ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, N, T, NFB)                
            ind_feat_enc_bottom = torch.transpose(ind_feat_enc_bottom, 1, 2).contiguous().view(B*T, N, NFB)
            # ind_feat_enc_bottom = self.spatial_transformer_encoder[1](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)
            ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, T, N, NFB)

        if self.use_res_connect:
            ind_feat_enc_bottom = ind_feat_enc_bottom + boxes_features_emb
            ind_feat_enc_upper = ind_feat_enc_upper + boxes_features_emb

        boxes_states = torch.cat([ind_feat_enc_upper, ind_feat_enc_bottom], dim=-1)
        NFS = NFB*2

        # Delete individual features of masked people
        if self.adapt_mask:
            boxes_states = boxes_states * (~(people_pad_mask.bool())).view(B, T, N, 1)
        if self.eval_only and (self.eval_mask_num != 0):
            boxes_states = boxes_states * (~(people_pad_mask.bool())).view(B, T, N, 1)

        # pooling individual features
        if self.people_pool_type == 'max':
            individual_feat, _ = torch.max(boxes_states, dim=1)
            group_feat, _ = torch.max(individual_feat, dim=1)
            group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)
        elif self.people_pool_type == 'mean':
            individual_feat = torch.mean(boxes_states, dim=1)
            group_feat = torch.mean(individual_feat, dim=1)
            group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)

        iar_inp_feat = group_feat_expand

        # add positional encoding to group features
        pos_enc_iar = self.pos_enc_iar.to(device=boxes_in.device)
        iar_loc_feat = torch.transpose(pos_enc_iar[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        iar_loc_feat = iar_loc_feat.view(B, T, N, NFS)
        if self.use_pos_cond:
            iar_inp_feat = iar_inp_feat + iar_loc_feat

        tem_enc_iar = self.tem_enc_iar.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        iar_tem_feat = tem_enc_iar[people_temp, :].view(B, T, N, NFS)
        if self.use_tmp_cond:
            iar_inp_feat = iar_inp_feat + iar_tem_feat

        ret_dic = {}
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, N*NFS)

        if self.training_stage == 3:
            if self.use_jae_loss:
                estimated_ja = self.fc_jae(iar_inp_feat).mean(dim=2)
                ret_dic['estimated_ja'] = estimated_ja
            else:
                activities = self.fc_activities(group_feat)
                ret_dic['activities'] = activities
                actions = self.fc_actions(individual_feat).reshape(B * N, -1)
                ret_dic['actions'] = actions
        else:
            if self.use_recon_loss:
                recon_features = self.fc_recon_with_gr(iar_inp_feat)
                ret_dic['recon_features'] = recon_features
                ret_dic['original_features'] = boxes_features
            if self.use_act_loss:
                actions_scores_with_gr = self.fc_actions_with_gr(iar_inp_feat)
                actions_scores_with_gr = torch.mean(actions_scores_with_gr, dim=1).reshape(B * N, -1)
                
                if self.use_gen_iar:
                    actions_scores_gen_iar = self.fc_actions_gen_iar(boxes_features_emb)
                    actions_scores_gen_iar = torch.mean(actions_scores_gen_iar, dim=1).reshape(B * N, -1)
                    actions_fused = actions_scores_gen_iar * self.gen_iar_ratio + actions_scores_with_gr * (1-self.gen_iar_ratio)
                    ret_dic['actions'] = actions_fused
                else:
                    ret_dic['actions'] = actions_scores_with_gr

        if self.use_exaplainable_mask:
            ret_dic['xai_person_mask'] = xai_person_mask

        return ret_dic

class GroupRelation_HiGCIN_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(GroupRelation_HiGCIN_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        # NFB = self.cfg.num_features_boxes
        NFB = D
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        # addtional parameters
        self.eval_only = self.cfg.eval_only
        self.use_random_mask = self.cfg.use_random_mask
        if self.use_random_mask:
            self.random_mask_type = self.cfg.random_mask_type
        self.use_ind_feat = self.cfg.use_ind_feat
        self.use_trans = self.cfg.use_trans
        self.trans_head_num = self.cfg.trans_head_num
        self.trans_layer_num = self.cfg.trans_layer_num
        self.people_pool_type = self.cfg.people_pool_type
        self.use_pos_cond = self.cfg.use_pos_cond
        self.use_tmp_cond = self.cfg.use_tmp_cond
        self.final_head_mid_num = self.cfg.final_head_mid_num
        self.use_recon_loss = self.cfg.use_recon_loss
        self.use_recon_diff_loss = self.cfg.use_recon_diff_loss
        self.use_act_loss = self.cfg.use_act_loss
        self.use_pose_loss = self.cfg.use_pose_loss
        self.use_old_act_rec = self.cfg.old_act_rec
        self.use_res_connect = self.cfg.use_res_connect
        self.use_gen_iar = self.cfg.use_gen_iar
        if self.use_gen_iar:
            self.gen_iar_ratio = self.cfg.gen_iar_ratio
        self.training_stage = self.cfg.training_stage
        self.ind_ext_feat_dim = K * K * D

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.person_avg_pool = nn.AvgPool2d((K**2, 1), stride = 1)
        self.BIM = CrossInferBlock(in_dim = D, Temporal = T, Spatial = K**2)
        self.PIM = CrossInferBlock(in_dim = D, Temporal = T, Spatial = N)
        self.dropout = nn.Dropout()
        self.fc_activities = nn.Linear(D, cfg.num_activities, bias = False)
        self.fc_actions = nn.Linear(D, cfg.num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        H, W = self.cfg.image_size
        self.pos_enc_iar = positionalencoding2d(NFB, H, W)
        self.final_head_mid = nn.Sequential()
        for i in range(self.final_head_mid_num):
            if i == 0:
                self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB, NFB))
            else:
                self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB, NFB))
            self.final_head_mid.add_module('relu%d' % i, nn.ReLU())

        if self.use_act_loss:
            self.fc_actions_with_gr = nn.Sequential(
                    self.final_head_mid,
                    nn.Linear(NFB, self.cfg.num_actions),
            )
        if self.use_recon_loss:
            self.fc_recon_with_gr = nn.Sequential(
                    self.final_head_mid,
                    nn.Linear(NFB, self.ind_ext_feat_dim),
            )

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        # self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # images_in, boxes_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        # images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.view(B, T, N, D, K*K)
        boxes_features_recon = boxes_features.view(B, T, N, K*K*D)
        boxes_features = boxes_features.permute(0, 2, 1, 4, 3).contiguous()
        boxes_features = boxes_features.view(B*N, T, K*K, D) # B*N, T, K*K, D

        # HiGCIN Inference
        boxes_features = self.BIM(boxes_features) # B*N, T, K*K, D
        boxes_features = self.person_avg_pool(boxes_features) # B*N, T, D
        boxes_features = boxes_features.view(B, N, T, D).contiguous().permute(0, 2, 1, 3) # B, T, N, D
        boxes_states = self.PIM(boxes_features) # B, T, N, D
        boxes_states = self.dropout(boxes_states)
        torch.cuda.empty_cache()
        NFS = D

        # pooling individual features
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)
        iar_inp_feat = group_feat_expand

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_iar = self.pos_enc_iar.to(device=boxes_in.device)
        iar_loc_feat = torch.transpose(pos_enc_iar[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        iar_loc_feat = iar_loc_feat.view(B, T, N, NFS)
        iar_inp_feat = iar_inp_feat + iar_loc_feat

        ret_dic = {}
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, N*NFS)

        if self.training_stage == 3:
            activities = self.fc_activities(group_feat)
            ret_dic['activities'] = activities
            actions = self.fc_actions(individual_feat).reshape(B * N, -1)
            ret_dic['actions'] = actions
        else:
            if self.use_recon_loss:
                recon_features = self.fc_recon_with_gr(iar_inp_feat)
                ret_dic['recon_features'] = recon_features
                ret_dic['original_features'] = boxes_features_recon
            if self.use_act_loss:
                actions_scores_with_gr = self.fc_actions_with_gr(iar_inp_feat)
                actions_scores_with_gr = torch.mean(actions_scores_with_gr, dim=1).reshape(B * N, -1)
                ret_dic['actions'] = actions_scores_with_gr

        return ret_dic

class GroupRelation_DIN_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(GroupRelation_DIN_volleyball, self).__init__()
        self.cfg=cfg
        
        T, N=self.cfg.num_frames, self.cfg.num_boxes
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG=self.cfg.num_graph

        # addtional parameters
        self.eval_only = self.cfg.eval_only
        self.use_random_mask = self.cfg.use_random_mask
        if self.use_random_mask:
            self.random_mask_type = self.cfg.random_mask_type
        self.use_ind_feat = self.cfg.use_ind_feat
        self.use_trans = self.cfg.use_trans
        self.trans_head_num = self.cfg.trans_head_num
        self.trans_layer_num = self.cfg.trans_layer_num
        self.people_pool_type = self.cfg.people_pool_type
        self.use_pos_cond = self.cfg.use_pos_cond
        self.use_tmp_cond = self.cfg.use_tmp_cond
        self.final_head_mid_num = self.cfg.final_head_mid_num
        self.use_recon_loss = self.cfg.use_recon_loss
        self.use_recon_diff_loss = self.cfg.use_recon_diff_loss
        self.use_act_loss = self.cfg.use_act_loss
        self.use_pose_loss = self.cfg.use_pose_loss
        self.use_old_act_rec = self.cfg.old_act_rec
        self.use_res_connect = self.cfg.use_res_connect
        self.use_gen_iar = self.cfg.use_gen_iar
        if self.use_gen_iar:
            self.gen_iar_ratio = self.cfg.gen_iar_ratio
        self.training_stage = self.cfg.training_stage
        self.ind_ext_feat_dim = K * K * D
        
        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained = True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained = True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained = True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False
        
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align = RoIAlign(*self.cfg.crop_size)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb_1 = nn.Linear(K*K*D,NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])
        
        
        #self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        if not self.cfg.hierarchical_inference:
            # self.DPI = Dynamic_Person_Inference(
            #     in_dim = in_dim,
            #     person_mat_shape = (10, 12),
            #     stride = cfg.stride,
            #     kernel_size = cfg.ST_kernel_size,
            #     dynamic_sampling=cfg.dynamic_sampling,
            #     sampling_ratio = cfg.sampling_ratio, # [1,2,4]
            #     group = cfg.group,
            #     scale_factor = cfg.scale_factor,
            #     beta_factor = cfg.beta_factor,
            #     parallel_inference = cfg.parallel_inference,
            #     cfg = cfg)
            self.DPI = Multi_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape = (10, 12),
                stride = cfg.stride,
                kernel_size = cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio = cfg.sampling_ratio, # [1,2,4]
                group = cfg.group,
                scale_factor = cfg.scale_factor,
                beta_factor = cfg.beta_factor,
                parallel_inference = cfg.parallel_inference,
                num_DIM = cfg.num_DIM,
                cfg = cfg)
            print_log(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        else:
            self.DPI = Hierarchical_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape=(10, 12),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                cfg = cfg,)
            print(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        self.dpi_nl = nn.LayerNorm([T, N, in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)


        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size = 1, stride = 1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
            self.fc_actions = nn.Linear(in_dim, self.cfg.num_actions)
        else:
            self.fc_activities=nn.Linear(NFG, self.cfg.num_activities)
            self.fc_actions = nn.Linear(NFG, self.cfg.num_actions)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.cfg.lite_dim:
            NFB = in_dim
        else:
            NFB = NFG

        H, W = self.cfg.image_size
        self.pos_enc_iar = positionalencoding2d(NFB, H, W)
        self.final_head_mid = nn.Sequential()
        for i in range(self.final_head_mid_num):
            if i == 0:
                self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB, NFB))
            else:
                self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB, NFB))
            self.final_head_mid.add_module('relu%d' % i, nn.ReLU())

        if self.use_act_loss:
            self.fc_actions_with_gr = nn.Sequential(
                    self.final_head_mid,
                    nn.Linear(NFB, self.cfg.num_actions),
            )
        if self.use_recon_loss:
            self.fc_recon_with_gr = nn.Sequential(
                    self.final_head_mid,
                    nn.Linear(NFB, self.ind_ext_feat_dim),
            )

                    
    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k,v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num +=1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num)+' parameters loaded for '+prefix)


    def forward(self,batch_data):
        # images_in, boxes_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        # images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]

        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4]==torch.Size([OH,OW])
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        
        
        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        boxes_features=boxes_features.reshape(B,T,N,-1)  #B,T,N, D*K*K
        boxes_features_recon = boxes_features.clone()

        # Embedding 
        boxes_features=self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features=self.nl_emb_1(boxes_features)
        boxes_features=F.relu(boxes_features, inplace = True)

        if self.cfg.lite_dim:
            boxes_features = boxes_features.permute(0, 3, 1, 2)
            boxes_features = self.point_conv(boxes_features)
            boxes_features = boxes_features.permute(0, 2, 3, 1)
            boxes_features = self.point_ln(boxes_features)
            boxes_features = F.relu(boxes_features, inplace = True)
        else:
            None

        # Dynamic graph inference
        # graph_boxes_features = self.DPI(boxes_features)
        graph_boxes_features, ft_infer_MAD = self.DPI(boxes_features)
        torch.cuda.empty_cache()


        if self.cfg.backbone == 'res18':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.dpi_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)
        elif self.cfg.backbone == 'vgg16':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace = True)
            boxes_states = self.dropout_global(boxes_states)
        elif self.cfg.backbone == 'inv3':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.dpi_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)

        # pooling individual features
        NFS = self.cfg.lite_dim
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)
        iar_inp_feat = group_feat_expand

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_iar = self.pos_enc_iar.to(device=boxes_in.device)
        iar_loc_feat = torch.transpose(pos_enc_iar[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        iar_loc_feat = iar_loc_feat.view(B, T, N, NFS)
        iar_inp_feat = iar_inp_feat + iar_loc_feat

        ret_dic = {}
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, N*NFS)

        if self.training_stage == 3:
            activities = self.fc_activities(group_feat)
            ret_dic['activities'] = activities
            actions = self.fc_actions(individual_feat).reshape(B * N, -1)
            ret_dic['actions'] = actions
        else:
            if self.use_recon_loss:
                recon_features = self.fc_recon_with_gr(iar_inp_feat)
                ret_dic['recon_features'] = recon_features
                ret_dic['original_features'] = boxes_features_recon
            if self.use_act_loss:
                actions_scores_with_gr = self.fc_actions_with_gr(iar_inp_feat)
                actions_scores_with_gr = torch.mean(actions_scores_with_gr, dim=1).reshape(B * N, -1)
                ret_dic['actions'] = actions_scores_with_gr

        return ret_dic

class GroupActivity_volleyball(nn.Module):
    """
    main module of GA recognition for the volleyball dataset
    """

    def __init__(self, cfg):
        super(GroupActivity_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        backbone_pretrain = True
        # backbone_pretrain = False
        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=backbone_pretrain)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=backbone_pretrain)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=backbone_pretrain)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=backbone_pretrain)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=backbone_pretrain)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])
        self.spatial_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1)
        ])

        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])
        self.spatial_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        # IAR with locations
        self.pos_enc_iar = positionalencoding2d(2*NFB, H, W)
        # self.fc_actions_with_gr = nn.Linear(NFB*2, self.cfg.num_actions)

        self.fc_actions = nn.Linear(NFB*2, self.cfg.num_actions)
        # self.fc_activities = nn.Linear(NFB*2, self.cfg.num_activities)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # images_in, boxes_in, images_person_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        ind_feat_set_upper = ind_feat_set.view(B*T, N, NFB)
        ind_feat_set_bottom = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)

        # generate padding masks for transformer
        people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()
        people_pad_mask_spatial = people_pad_mask.view(B*T, N)
        people_pad_mask_temporal = torch.transpose(people_pad_mask, 1, 2).contiguous().view(B*N, T)

        # transformer encoder
        ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_enc_upper)
        ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
        ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

        # bottom branch
        ind_feat_enc_bottom = self.temporal_transformer_encoder[1](ind_feat_set_bottom)
        ind_feat_enc_bottom_res = self.temporal_transformer_mlp[0](ind_feat_enc_bottom)
        ind_feat_enc_bottom = ind_feat_set_bottom + ind_feat_enc_bottom_res
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, N, T, NFB)                
        ind_feat_enc_bottom = torch.transpose(ind_feat_enc_bottom, 1, 2).contiguous().view(B*T, N, NFB)
        ind_feat_enc_bottom = self.spatial_transformer_encoder[1](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, T, N, NFB)

        boxes_states = torch.cat([ind_feat_enc_upper, ind_feat_enc_bottom], dim=-1)
        NFS = NFB*2

        # ============================= general group activity recognition
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Predict activities
        # boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        # boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFS)
        # activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        # activities_scores = activities_scores.reshape(B, T, -1)
        # activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)
        # ============================= general group activity recognition

        # pooling individual features
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)

        ret_dic = {}
        ret_dic['actions'] = actions_scores
        # ret_dic['activities'] = activities_scores
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, N*NFS)

        return ret_dic

class DualAI_volleyball(nn.Module):
    """
    main module of GA recognition for the volleyball dataset
    """

    def __init__(self, cfg):
        super(DualAI_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])
        self.spatial_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1)
        ])
        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])
        self.spatial_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        self.fc_actions = nn.Linear(NFB*2, self.cfg.num_actions)
        # self.fc_activities = nn.Linear(NFB*2, self.cfg.num_activities)
        self.fc_activities = nn.Sequential(
                nn.Linear(NFB*2, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, cfg.num_activities),
                )

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        if len(batch_data) == 2:
            images_in, boxes_in = batch_data
        else:
            images_in = batch_data['images_in']
            boxes_in = batch_data['boxes_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        ind_feat_set_upper = ind_feat_set.view(B*T, N, NFB)
        ind_feat_set_bottom = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)

        # generate padding masks for transformer
        people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()
        people_pad_mask_spatial = people_pad_mask.view(B*T, N)
        people_pad_mask_temporal = torch.transpose(people_pad_mask, 1, 2).contiguous().view(B*N, T)

        # transformer encoder
        ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_enc_upper)
        ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
        ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

        # bottom branch
        ind_feat_enc_bottom = self.temporal_transformer_encoder[1](ind_feat_set_bottom)
        ind_feat_enc_bottom_res = self.temporal_transformer_mlp[0](ind_feat_enc_bottom)
        ind_feat_enc_bottom = ind_feat_set_bottom + ind_feat_enc_bottom_res
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, N, T, NFB)                
        ind_feat_enc_bottom = torch.transpose(ind_feat_enc_bottom, 1, 2).contiguous().view(B*T, N, NFB)
        ind_feat_enc_bottom = self.spatial_transformer_encoder[1](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, T, N, NFB)

        boxes_states = torch.cat([ind_feat_enc_upper, ind_feat_enc_bottom], dim=-1)
        NFS = NFB*2

        # ============================= general group activity recognition
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFS)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)
        # ============================= general group activity recognition

        ret_dic = {}
        ret_dic['actions'] = actions_scores
        ret_dic['activities'] = activities_scores

        return ret_dic

class PersonAction_volleyball(nn.Module):

    def __init__(self, cfg):
        super(PersonAction_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])
        self.spatial_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1)
        ])

        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])
        self.spatial_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        # IAR with locations
        self.pos_enc_iar = positionalencoding2d(2*NFB, H, W)
        # self.fc_actions_with_gr = nn.Linear(NFB*2, self.cfg.num_actions)

        self.fc_actions = nn.Linear(NFB*2, self.cfg.num_actions)
        # self.fc_activities = nn.Linear(NFB*2, self.cfg.num_activities)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # images_in, boxes_in, images_person_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        ind_feat_set_upper = ind_feat_set.view(B*T, N, NFB)
        ind_feat_set_bottom = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)

        # generate padding masks for transformer
        people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()
        people_pad_mask_spatial = people_pad_mask.view(B*T, N)
        people_pad_mask_temporal = torch.transpose(people_pad_mask, 1, 2).contiguous().view(B*N, T)

        # transformer encoder
        ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_enc_upper)
        ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
        ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

        # bottom branch
        ind_feat_enc_bottom = self.temporal_transformer_encoder[1](ind_feat_set_bottom)
        ind_feat_enc_bottom_res = self.temporal_transformer_mlp[0](ind_feat_enc_bottom)
        ind_feat_enc_bottom = ind_feat_set_bottom + ind_feat_enc_bottom_res
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, N, T, NFB)                
        ind_feat_enc_bottom = torch.transpose(ind_feat_enc_bottom, 1, 2).contiguous().view(B*T, N, NFB)
        ind_feat_enc_bottom = self.spatial_transformer_encoder[1](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, T, N, NFB)

        boxes_states = torch.cat([ind_feat_enc_upper, ind_feat_enc_bottom], dim=-1)
        NFS = NFB*2

        # ============================= general group activity recognition
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        # ============================= general group activity recognition

        # pooling individual features
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)

        ret_dic = {}
        ret_dic['pseudo_scores'] = actions_scores
        ret_dic['person_features'] = individual_feat.view(B, N, NFS)

        return ret_dic

class PersonActionSigleBranch_volleyball(nn.Module):

    def __init__(self, cfg):
        super(PersonActionSigleBranch_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])
        self.spatial_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])

        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])
        self.spatial_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        # IAR with locations
        self.pos_enc_iar = positionalencoding2d(NFB, H, W)
        # self.fc_actions_with_gr = nn.Linear(NFB*2, self.cfg.num_actions)

        self.fc_actions = nn.Linear(NFB, self.cfg.num_actions)
        # self.fc_activities = nn.Linear(NFB*2, self.cfg.num_activities)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # images_in, boxes_in, images_person_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        ind_feat_set_upper = ind_feat_set.view(B*T, N, NFB)
        ind_feat_set_bottom = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)

        # generate padding masks for transformer
        people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()
        people_pad_mask_spatial = people_pad_mask.view(B*T, N)
        people_pad_mask_temporal = torch.transpose(people_pad_mask, 1, 2).contiguous().view(B*N, T)

        # transformer encoder
        ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_enc_upper)
        ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
        ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

        boxes_states = torch.cat([ind_feat_enc_upper], dim=-1)
        NFS = NFB

        # ============================= general group activity recognition
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        # ============================= general group activity recognition

        # pooling individual features
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)

        ret_dic = {}
        ret_dic['pseudo_scores'] = actions_scores
        ret_dic['person_features'] = individual_feat.view(B, N, NFS)

        return ret_dic

class PersonActionSigleBranchTemporal_volleyball(nn.Module):

    def __init__(self, cfg):
        super(PersonActionSigleBranchTemporal_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])

        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        # IAR with locations
        self.pos_enc_iar = positionalencoding2d(NFB, H, W)
        # self.fc_actions_with_gr = nn.Linear(NFB*2, self.cfg.num_actions)

        self.fc_actions = nn.Linear(NFB, self.cfg.num_actions)
        # self.fc_activities = nn.Linear(NFB*2, self.cfg.num_activities)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # images_in, boxes_in, images_person_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        ind_feat_set = ind_feat_set.view(B, T, N, NFB)
        ind_feat_set = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)
        ind_feat_set = self.temporal_transformer_encoder[0](ind_feat_set)                
        ind_feat_set = ind_feat_set.view(B, N, T, NFB)
        ind_feat_set = torch.transpose(ind_feat_set, 1, 2)
        boxes_states = torch.cat([ind_feat_set], dim=-1)
        NFS = NFB

        # ============================= general group activity recognition
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        # ============================= general group activity recognition

        # pooling individual features
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)

        ret_dic = {}
        ret_dic['pseudo_scores'] = actions_scores
        ret_dic['person_features'] = individual_feat.view(B, N, NFS)

        return ret_dic