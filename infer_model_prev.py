from backbone.backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
import collections
import sys

class GroupRelationIdentity_volleyball(nn.Module):
    """
    main module of GR learning for the volleyball dataset
    """

    def __init__(self, cfg):
        super(GroupRelationIdentity_volleyball, self).__init__()
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
        elif cfg.backbone == 'vgg19_flat':
            self.backbone = MyVGG19Flat(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.use_ind_feat_crop = self.cfg.use_ind_feat_crop

        if self.use_ind_feat_crop == 'roi_multi':
            out_feat_size = 12800
            self.hr_inp_dim = [12800, 1012, 512, 1012]
            self.hr_out_dim = [1012, 512, 1012, 12800]
        elif self.use_ind_feat_crop == 'crop_single':
            out_feat_size = 4096
            self.hr_inp_dim = [4096, 256, 128, 256]
            self.hr_out_dim = [256, 128, 256, 4096]

        self.use_loc_feat_prev = self.cfg.use_loc_feat_prev
        if self.use_loc_feat_prev:
            self.pos_enc_ind = positionalencoding2d(out_feat_size, H, W)
            self.tem_enc_ind = positionalencoding1d(out_feat_size, 100)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
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
        PH, PW = self.cfg.person_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4
        images_person_in_flat = torch.reshape(images_person_in, (B * T * N, 3, PH, PW))  # B*T, 3, H, W

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        if self.use_ind_feat_crop in ['roi_multi']:
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
        elif self.use_ind_feat_crop in ['crop_single']:
            # Use backbone to extract features of images_persons_in
            # Pre-precess first
            images_person_in_flat = prep_images(images_person_in_flat)
            outputs = self.backbone(images_person_in_flat)
            boxes_features = outputs[-1].reshape(B, T, N, -1)  # B,T,N, D*K*K
        else:
            assert False, 'use_ind_feat_crop error'

        if self.use_loc_feat_prev:
            # encode position infromation
            boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
            boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
            boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
            boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
            pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
            ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
            ind_loc_feat = ind_loc_feat.view(B, T, N, -1)

            # encode temporal infromation
            tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
            people_tem = torch.arange(T).to(device=boxes_in.device).long()
            people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
            ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, -1)

        # generate individual features
        if self.use_loc_feat_prev:
            ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        else:
            ind_feat_set = boxes_features

        # pooling individual features
        individual_feat, _ = torch.max(ind_feat_set, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)

        ret_dic = {}
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, -1)

        return ret_dic

class GroupRelationAutoEncoder_volleyball(nn.Module):
    """
    main module of GR learning for the volleyball dataset
    """

    def __init__(self, cfg):
        super(GroupRelationAutoEncoder_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        dataset_name = self.cfg.dataset_name

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'vgg19_flat':
            self.backbone = MyVGG19Flat(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.use_ind_feat_crop = self.cfg.use_ind_feat_crop

        if self.use_ind_feat_crop == 'roi_multi':
            if dataset_name == 'volleyball':
                out_feat_size = 12800
            elif dataset_name == 'collective':
                out_feat_size = 26400
            self.ae_inp_dim = [out_feat_size, 2048, 1024, 2048]
            self.ae_out_dim = [2048, 1024, 2048, out_feat_size]
        elif self.use_ind_feat_crop == 'crop_single':
            out_feat_size = 4096
            self.ae_inp_dim = [4096, 256, 128, 256]
            self.ae_out_dim = [256, 128, 256, 4096]

        self.use_loc_feat_prev = self.cfg.use_loc_feat_prev
        if self.use_loc_feat_prev:
            self.pos_enc_ind = positionalencoding2d(out_feat_size, H, W)
            self.tem_enc_ind = positionalencoding1d(out_feat_size, 100)

        # encoder
        self.encoder = nn.Sequential(
                nn.Linear(self.ae_inp_dim[0], self.ae_out_dim[0]),
                nn.LayerNorm([self.ae_out_dim[0]]),
                nn.ReLU(),
                nn.Linear(self.ae_inp_dim[1], self.ae_out_dim[1]),
                nn.LayerNorm([self.ae_out_dim[1]]),
        )

        # decoder
        self.decoder = nn.Sequential(
                nn.Linear(self.ae_inp_dim[2], self.ae_out_dim[2]),
                nn.LayerNorm([self.ae_out_dim[2]]),
                nn.ReLU(),
                nn.Linear(self.ae_inp_dim[3], self.ae_out_dim[3]),
        )

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
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
        PH, PW = self.cfg.person_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4
        images_person_in_flat = torch.reshape(images_person_in, (B * T * N, 3, PH, PW))  # B*T, 3, H, W

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        if self.use_ind_feat_crop in ['roi_multi']:
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
        elif self.use_ind_feat_crop in ['crop_single']:
            # Use backbone to extract features of images_persons_in
            # Pre-precess first
            images_person_in_flat = prep_images(images_person_in_flat)
            outputs = self.backbone(images_person_in_flat)
            boxes_features = outputs[-1].reshape(B, T, N, -1)  # B,T,N, D*K*K
        else:
            assert False, 'use_ind_feat_crop error'

        if self.use_loc_feat_prev:
            # encode position infromation
            boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
            boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
            boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
            boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
            pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
            ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
            ind_loc_feat = ind_loc_feat.view(B, T, N, -1)

            # encode temporal infromation
            tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
            people_tem = torch.arange(T).to(device=boxes_in.device).long()
            people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
            ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, -1)

        # generate individual features
        if self.use_loc_feat_prev:
            ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        else:
            ind_feat_set = boxes_features

        ind_feat_set = boxes_features
        ind_feat_set_latent = self.encoder(ind_feat_set)
        ind_feat_set_recon = self.decoder(ind_feat_set_latent)

        # pooling individual features
        individual_feat, _ = torch.max(ind_feat_set_latent, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)

        ret_dic = {}
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, -1)
        ret_dic['original_features'] = ind_feat_set
        ret_dic['recon_features'] = ind_feat_set_recon

        return ret_dic

class GroupRelationHRN_volleyball(nn.Module):
    """
    main module of GR learning for the volleyball dataset
    """

    def __init__(self, cfg):
        super(GroupRelationHRN_volleyball, self).__init__()
        self.cfg = cfg
        self.exp_note = self.cfg.exp_note

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        self.dataset_name = self.cfg.dataset_name

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'vgg19_flat':
            self.backbone = MyVGG19Flat(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.use_ind_feat_crop = self.cfg.use_ind_feat_crop

        if self.use_ind_feat_crop == 'roi_multi':
            if self.dataset_name == 'volleyball':
                out_feat_size = 12800
            elif self.dataset_name == 'collective':
                out_feat_size = 26400
            self.hr_inp_dim = [out_feat_size, 1012, 512, 1012]
            self.hr_out_dim = [1012, 512, 1012, out_feat_size]
        elif self.use_ind_feat_crop == 'crop_single':
            out_feat_size = 4096
            self.hr_inp_dim = [4096, 256, 128, 256]
            self.hr_out_dim = [256, 128, 256, 4096]

        self.use_loc_feat_prev = self.cfg.use_loc_feat_prev
        if self.use_loc_feat_prev:
            self.pos_enc_ind = positionalencoding2d(out_feat_size, H, W)
            self.tem_enc_ind = positionalencoding1d(out_feat_size, 100)

        # relation network
        self.vol_flag = not 'crop' in self.exp_note
        self.cad_flag = 'crop' in self.exp_note
        if (self.dataset_name=='volleyball' and self.vol_flag) or (self.dataset_name=='collective' and self.cad_flag):
            self.hrn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(self.hr_inp_dim[i]*2, self.hr_out_dim[i]),
                    nn.ReLU(),
                )
                for i in range(len(self.hr_inp_dim))]
            )
        else:
            self.hrn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(self.hr_inp_dim[i]*2, self.hr_out_dim[i]),
                    nn.ReLU(),
                )
                for i in range(len(self.hr_inp_dim)-1)]
            )
            self.hrn_layer_final = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(self.hr_inp_dim[-1]*2, self.hr_out_dim[-1]),
                    nn.Sigmoid(),
                    )

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
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
        PH, PW = self.cfg.person_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4
        images_person_in_flat = torch.reshape(images_person_in, (B * T * N, 3, PH, PW))  # B*T, 3, H, W

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        if self.use_ind_feat_crop in ['roi_multi']:
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
        elif self.use_ind_feat_crop in ['crop_single']:
            # Use backbone to extract features of images_persons_in
            # Pre-precess first
            images_person_in_flat = prep_images(images_person_in_flat)
            outputs = self.backbone(images_person_in_flat)
            boxes_features = outputs[-1].reshape(B, T, N, -1)  # B,T,N, D*K*K
        else:
            assert False, 'use_ind_feat_crop error'

        if self.use_loc_feat_prev:
            # encode position infromation
            boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
            boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
            boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
            boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
            pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
            ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
            ind_loc_feat = ind_loc_feat.view(B, T, N, -1)

            # encode temporal infromation
            tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
            people_tem = torch.arange(T).to(device=boxes_in.device).long()
            people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
            ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, -1)

        # generate individual features
        if self.use_loc_feat_prev:
            ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        else:
            ind_feat_set = boxes_features

        # process by hierarchical relation network
        ind_feat_set = boxes_features
        ind_feat_set_hr = ind_feat_set.view(B*T, N, 1, self.hr_inp_dim[0])
        for hr_layer_idx in range(len(self.hr_inp_dim)):
            ind_feat_set_hr_col = ind_feat_set_hr.view(B*T, N, 1, self.hr_inp_dim[hr_layer_idx])
            ind_feat_set_hr_row = ind_feat_set_hr.view(B*T, 1, N, self.hr_inp_dim[hr_layer_idx])
            ind_feat_set_hr_col_expand = ind_feat_set_hr_col.expand(B*T, N, N, self.hr_inp_dim[hr_layer_idx])
            ind_feat_set_hr_row_expand = ind_feat_set_hr_row.expand(B*T, N, N, self.hr_inp_dim[hr_layer_idx])
            ind_feat_set_hr_expand = torch.cat([ind_feat_set_hr_col_expand, ind_feat_set_hr_row_expand], dim=-1) 
            
            if (self.dataset_name=='volleyball' and self.vol_flag) or (self.dataset_name=='collective' and self.cad_flag):
                ind_feat_set_hr = torch.sum(self.hrn_layers[hr_layer_idx](ind_feat_set_hr_expand), dim=-2)
            else:
                if hr_layer_idx == (len(self.hr_inp_dim)-1):
                    ind_feat_set_hr = torch.sum(self.hrn_layer_final(ind_feat_set_hr_expand), dim=-2)
                else:
                    ind_feat_set_hr = torch.sum(self.hrn_layers[hr_layer_idx](ind_feat_set_hr_expand), dim=-2)
            
            if hr_layer_idx == 1:
                ind_feat_set_hr_latent = ind_feat_set_hr
        ind_feat_set = ind_feat_set.view(B, T, N, self.hr_inp_dim[0])
        ind_feat_set_latent = ind_feat_set_hr_latent.view(B, T, N, self.hr_out_dim[1])
        ind_feat_set_recon = ind_feat_set_hr.view(B, T, N, self.hr_out_dim[-1])

        # pooling individual features
        individual_feat, _ = torch.max(ind_feat_set_latent, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)

        ret_dic = {}
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, -1)
        ret_dic['original_features'] = ind_feat_set
        ret_dic['recon_features'] = ind_feat_set_recon

        return ret_dic
