'''
    Train the model with person action/appearance features.
'''

import torch
import torch.optim as optim

import time
import random
import os
import sys
import wandb
from tqdm import tqdm
from collections import OrderedDict
from fast_pytorch_kmeans import KMeans as KMeansGPU

from config import *
from volleyball import *
from collective import *
from dataset import *
from infer_model import *
from infer_model_prev import *
from infer_model_original import *
from base_model import *
from utils import *

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)

    # Save config parameters
    cfg_save_path = os.path.join(cfg.result_path, 'cfg.pickle')
    with open(cfg_save_path, "wb") as f:
        pickle.dump(cfg, f)
    
    # Reading dataset
    training_set,validation_set=return_dataset(cfg)
    
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 8, # 4,
    }
    training_loader=data.DataLoader(training_set,**params)
    
    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)
    
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    torch.cuda.manual_seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    basenet_list={'volleyball':Basenet_volleyball, 'collective':Basenet_collective}
    gcnnet_list={
                 'group_activity_volleyball':GroupActivity_volleyball,
                 'group_relation_volleyball':GroupRelation_volleyball,
                 'group_relation_higcin_volleyball':GroupRelation_HiGCIN_volleyball,
                 'group_relation_din_volleyball':GroupRelation_DIN_volleyball,
                 'group_relation_ident_volleyball':GroupRelationIdentity_volleyball,
                 'group_relation_ae_volleyball':GroupRelationAutoEncoder_volleyball,
                 'group_relation_hrn_volleyball':GroupRelationHRN_volleyball,
                 'group_activity_collective':GroupActivity_volleyball,
                 'group_relation_collective':GroupRelation_volleyball,
                 'group_relation_higcin_collective':GroupRelation_HiGCIN_volleyball,
                 'group_relation_din_collective':GroupRelation_DIN_volleyball,
                 'group_relation_ident_collective':GroupRelationIdentity_volleyball,
                 'group_relation_ae_collective':GroupRelationAutoEncoder_volleyball,
                 'group_relation_hrn_collective':GroupRelationHRN_volleyball,
                 'dynamic_volleyball':Dynamic_volleyball,
                 'dynamic_collective':Dynamic_volleyball,
                 'higcin_volleyball':HiGCIN_volleyball,
                 'higcin_collective':HiGCIN_volleyball,
                 'person_action_recognizor':PersonAction_volleyball,
                 'single_branch_transformer':PersonActionSigleBranch_volleyball,
                 'single_branch_transformer_wo_spatial':PersonActionSigleBranchTemporal_volleyball,
                 }
    
    # build main GAFL network
    if cfg.training_stage==1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
    elif cfg.training_stage==2:
        GCNnet = gcnnet_list[cfg.inference_module_name]
        model = GCNnet(cfg)
        # Load backbone
        if cfg.load_backbone_stage2:
            model.loadmodel(cfg.stage1_model_path)
        elif cfg.load_stage2model:
            # if cfg.use_multi_gpu:
            #     model = nn.DataParallel(model)
            state = torch.load(cfg.stage2model)
            model.load_state_dict(state['state_dict'])
            print_log(cfg.log_path, 'Loading stage2 model: ' + cfg.stage2model)
        else:
            print_log(cfg.log_path, 'Not loading stage1 or stage2 model.')
    else:
        assert(False)

    # move models to gpu
    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)

    model=model.to(device=device)

    # set mode of models    
    model.train()
    if cfg.set_bn_eval:
        model.apply(set_bn_eval)

    # set parameters to be optimized
    optimizer_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    # build sub action recognition network (full scratch)
    if cfg.mode == 'PAC_DC':
        model_pac_net = gcnnet_list[cfg.pseudo_classifier_mode](cfg)

        if cfg.use_pseudo_act_gt:
            model_pac_net.loadmodel(cfg.stage1_model_path)
            print_log(cfg.log_path, 'Loading stage1 model for recognizor: ' + cfg.stage1_model_path)

        model_pac_net = nn.DataParallel(model_pac_net)
        model_pac_net = model_pac_net.to(device=device)
        model_pac_net.train()
        optimizer_params += list(filter(lambda p: p.requires_grad, model_pac_net.parameters()))
    
    # build sub action recognition network (pretrained)
    if cfg.mode == 'PAC_REC':
        model_act_recognizer = DualAI_volleyball(cfg)
        state_act_recognizer = torch.load(cfg.act_recognizer_path)['state_dict']
        new_state_act_recognizer=OrderedDict()
        for k, v in state_act_recognizer.items():
            name = k[7:] 
            new_state_act_recognizer[name] = v
        model_act_recognizer.load_state_dict(new_state_act_recognizer)
        model_act_recognizer = model_act_recognizer.to(device=device)
        model_act_recognizer.eval()

    optimizer=optim.Adam(optimizer_params, lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)

    models = {'model': model}
    if cfg.mode == 'PAC_REC':
        models['model_act_recognizer'] = model_act_recognizer
    if cfg.mode == 'PAC_DC':
        models['model_pac_net'] = model_pac_net

    train_list={'volleyball':train_volleyball, 'collective':train_collective}
    test_list={'volleyball':test_volleyball, 'collective':test_collective}
    train=train_list[cfg.dataset_name]
    test=test_list[cfg.dataset_name]
    
    if cfg.test_before_train:
        test_info=test(validation_loader, models, device, 0, cfg)
        print(test_info)

    # Training iteration
    best_result = {'epoch':0, 'activities_acc':0, 'actions_acc':0, 'loss':100000000000000}
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
            
        # One epoch of forward and backward
        train_info=train(training_loader, models, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)
        for wandb_loss_name in cfg.wandb_loss_list:
            wandb.log({f"Train {wandb_loss_name}": train_info[wandb_loss_name]}, step=epoch)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info=test(validation_loader, models, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info)
            for wandb_loss_name in cfg.wandb_loss_list:
                wandb.log({f"Test {wandb_loss_name}": test_info[wandb_loss_name]}, step=epoch)

            # if test_info['activities_acc']>best_result['activities_acc']:
            # if test_info['actions_acc']>best_result['actions_acc']:
            if test_info['loss']<best_result['loss']:
                best_result=test_info
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if cfg.mode == 'PAC_DC':
                    state['state_dict_pac_net'] = model_pac_net.state_dict()
                torch.save(state, cfg.result_path+'/best_model.pth')
                print('Best model saved.')
            print_log(cfg.log_path, 
                    #   'Best group activity accuracy: %.2f%% at epoch #%d.'%(best_result['activities_acc'], best_result['epoch']))
                    #   'Best individual action accuracy: %.2f%% at epoch #%d.'%(best_result['actions_acc'], best_result['epoch']))
                      'Best loss: %.2f%% at epoch #%d.'%(best_result['loss'], best_result['epoch']))

            # Save model
            if cfg.training_stage==2:
                # None
                # if test_info['activities_acc'] > 93.1:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if cfg.mode == 'PAC_DC':
                    state['state_dict_pac_net'] = model_pac_net.state_dict()
                # filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
                # filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['actions_acc'])
                filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['loss'])
                torch.save(state, filepath)
                print('model saved to:',filepath)
            elif cfg.training_stage==1:
                # if test_info['activities_acc'] == best_result['activities_acc']:
                # if test_info['actions_acc'] == best_result['actions_acc']:
                if test_info['loss'] == best_result['loss']:
                    for m in model.modules():
                        if isinstance(m, Basenet):
                            # filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
                            # filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['actions_acc'])
                            filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['loss'])
                            m.savemodel(filepath)
    #                         print('model saved to:',filepath)
            else:
                assert False
   
def train_volleyball(data_loader, models, device, optimizer, epoch, cfg):
    train_with_action = False
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    loss_act_meter=AverageMeter()
    loss_recon_meter=AverageMeter()
    loss_pseudo_rec_meter=AverageMeter()
    loss_cluster_balance_meter=AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)
    epoch_timer=Timer()
    model = models['model']

    # generate pseudo labels from person perfeatures with k-means
    if cfg.mode == 'PAC_DC':
        print('Obtain features for clustering...')
        model_pac_net = models['model_pac_net']
        
        batch_idx = -1
        for batch_data in tqdm(data_loader):
            batch_idx += 1
            with torch.no_grad():
                for key in range(len(batch_data)):
                    if torch.is_tensor(batch_data[key]):
                        batch_data[key] = batch_data[key].to(device=device)
                batch_size=batch_data[0].shape[0]
                num_frames=batch_data[0].shape[1]
                input_data = {}
                input_data['images_in'] = batch_data[0]
                input_data['boxes_in'] = batch_data[1]
                input_data['actions_in'] = batch_data[2]
                input_data['images_person_in'] = batch_data[4]

                # obtain person features
                ret_pac_net = model_pac_net(input_data)
                person_features_pac_net = ret_pac_net['person_features']
                person_features_pac_net_dim = person_features_pac_net.shape[-1]
                if batch_idx == 0:
                    person_features_pac_net_all = torch.zeros(len(data_loader), cfg.batch_size, cfg.num_boxes, person_features_pac_net_dim).to(device=device)
                person_features_pac_net_all[batch_idx, :batch_size] = person_features_pac_net.detach().reshape(batch_size, cfg.num_boxes, -1)
        person_features_pac_net_all = person_features_pac_net_all.reshape(len(data_loader)*cfg.batch_size*cfg.num_boxes, -1)

        print('Clustering in processing...')
        kmeans_gpu = KMeansGPU(n_clusters=cfg.num_actions, mode='euclidean', verbose=0)
        pseudo_action_labels = kmeans_gpu.fit_predict(person_features_pac_net_all)
        pseudo_action_labels = pseudo_action_labels.reshape(len(data_loader), cfg.batch_size*cfg.num_boxes)
        pseudo_action_labels_weight = torch.zeros(cfg.num_actions).to(device=device)
        for i in range(cfg.num_actions):
            pseudo_action_labels_weight[i] = torch.sum(torch.eq(pseudo_action_labels, i).float())
        print('Clustering results:', pseudo_action_labels_weight)
        pseudo_action_labels_weight_inv = (1 / pseudo_action_labels_weight)
        pseudo_action_labels_weight_inv = pseudo_action_labels_weight_inv / torch.sum(pseudo_action_labels_weight_inv)
        print('pseudo_action_labels_weight_inv:', pseudo_action_labels_weight_inv)

    for batch_idx, batch_data in enumerate(tqdm(data_loader)):
        if batch_idx % 100 == 0 and batch_idx > 0:
            print('Training in processing {}/{}, Loss: {:.4f}'.format(batch_idx, len(data_loader), loss_meter.avg))

        model.train()
        if cfg.set_bn_eval:
            model.apply(set_bn_eval)
    
        # prepare batch data
        for key in range(len(batch_data)):
            if torch.is_tensor(batch_data[key]):
                batch_data[key] = batch_data[key].to(device=device)

        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
        activities_in=batch_data[3].reshape((batch_size,num_frames))

        actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
        activities_in=activities_in[:,0].reshape((batch_size,))

        # forward
        input_data = {}
        input_data['images_in'] = batch_data[0]
        input_data['boxes_in'] = batch_data[1]
        input_data['images_person_in'] = batch_data[4]
        input_data['actions_in'] = batch_data[2]

        # define list for various losses
        loss_list = []

        # recognize actions
        if cfg.mode == 'PAC_REC':
            ret_rec_act = models['model_act_recognizer'](input_data)
            rec_act_labels = torch.argmax(ret_rec_act['actions'], dim=1)
            input_data['actions_in'] = rec_act_labels
        
        # predict pseudo actions
        if cfg.mode == 'PAC_DC':
            model_pac_net.train()
            ret_pac_net = model_pac_net(input_data)
            action_scores_pac_net = ret_pac_net['pseudo_scores']
            action_scores_pac_net_sm = F.softmax(action_scores_pac_net, dim=-1)
            action_scores_pac_net = action_scores_pac_net.reshape(batch_size*cfg.num_boxes, -1)
            if cfg.use_pseudo_act_gt:
                actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
                pseudo_actions_loss = F.cross_entropy(action_scores_pac_net, actions_in, weight=actions_weights)
            else:
                if cfg.use_pseudo_act_loss_weight:
                    pseudo_actions_loss = F.cross_entropy(action_scores_pac_net, pseudo_action_labels[batch_idx][:batch_size*cfg.num_boxes], weight=pseudo_action_labels_weight_inv)
                else:
                    pseudo_actions_loss = F.cross_entropy(action_scores_pac_net, pseudo_action_labels[batch_idx][:batch_size*cfg.num_boxes])
            loss_list.append(pseudo_actions_loss)
            loss_pseudo_rec_meter.update(pseudo_actions_loss.item(), batch_size)
            
            # clustering balance loss
            if cfg.use_cluster_balance_loss:
                action_scores_pac_net_sm_mean = torch.mean(action_scores_pac_net_sm, dim=0)
                action_scores_pac_net_sm_mean_expand = action_scores_pac_net_sm_mean.unsqueeze(1).expand(cfg.num_actions, cfg.num_actions)
                action_scores_pac_net_sm_prior = F.softmax(torch.tensor(cfg.actions_weights, dtype=torch.float32).to(device=device), dim=-1)
                action_scores_pac_net_sm_prior_expand = action_scores_pac_net_sm_prior.unsqueeze(0).expand(cfg.num_actions, cfg.num_actions)
                action_scores_cost_matrix = torch.abs(action_scores_pac_net_sm_prior_expand - action_scores_pac_net_sm_mean_expand)
                action_scores_cost_matrix = torch.zeros(cfg.num_actions, cfg.num_actions).to(device=device)
                for i in range(cfg.num_actions):
                    for j in range(cfg.num_actions):
                        action_scores_cost_matrix[i][j] = torch.norm(action_scores_pac_net_sm_prior[i] - action_scores_pac_net_sm_mean[j])
                action_scores_pac_net_sm_prior_sort = torch.zeros(cfg.num_actions).to(device=device)
                hungarian = Hungarian(action_scores_cost_matrix.detach().cpu().numpy())
                hungarian.calculate()
                hungarian_results = hungarian.get_results()
                for i in range(cfg.num_actions):
                    mean_idx, prior_idx = hungarian_results[i]
                    action_scores_pac_net_sm_prior_sort[mean_idx] = action_scores_pac_net_sm_prior[prior_idx]
                cluster_balance_loss = F.mse_loss(action_scores_pac_net_sm_mean, action_scores_pac_net_sm_prior_sort) * 10e2
                loss_list.append(cluster_balance_loss)
                loss_cluster_balance_meter.update(cluster_balance_loss.item(), batch_size)

        # obtain group features        
        ret= model(input_data)

        # Predict activities
        if 'activities' in list(ret.keys()):
            activities_scores = ret['activities']
            activities_loss = F.cross_entropy(activities_scores,activities_in)
            loss_list.append(activities_loss)
            activities_labels = torch.argmax(activities_scores,dim=1)
            activities_correct = torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]
            activities_meter.update(activities_accuracy, activities_scores.shape[0])
            activities_conf.add(activities_labels, activities_in)

        # Predict actions
        if 'actions' in list(ret.keys()):
            actions_scores = ret['actions']
            actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
            if cfg.mode == 'PAC_DC':
                actions_loss = F.cross_entropy(actions_scores, action_scores_pac_net_sm) * cfg.actions_loss_weight
            else:
                actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights) * cfg.actions_loss_weight
            loss_list.append(actions_loss)
            actions_labels = torch.argmax(actions_scores, dim=1)
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            actions_conf.add(actions_labels, actions_in)
            loss_act_meter.update(actions_loss.item(), batch_size)

        # Predict features
        if 'recon_features' in list(ret.keys()):
            recon_features = ret['recon_features']
            original_features = ret['original_features']
            recon_loss = F.mse_loss(recon_features, original_features)
            loss_list.append(recon_loss)
            loss_recon_meter.update(recon_loss.item(), batch_size)

        if 'halting' in list(ret.keys()):
            loss_list.append(ret['halting']*cfg.halting_penalty)

        # print(loss_list)
        total_loss = sum(loss_list)
        loss_meter.update(total_loss.item(), batch_size)
        # print(f'===> [{epoch}] [{batch_idx}/{len(data_loader)}]: {total_loss.item():.4f}')

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        # Test max_clip_norm
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'loss_pseudo_rec':loss_pseudo_rec_meter.avg,
        'loss_cluster_balance':loss_cluster_balance_meter.avg,
        'loss_act':loss_act_meter.avg,
        'loss_recon':loss_recon_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'activities_conf':activities_conf.value(),
        'activities_MPCA':MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
    }      

    return train_info
  
def test_volleyball(data_loader, models, device, epoch, cfg):
    train_with_action = False
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    loss_act_meter=AverageMeter()
    loss_recon_meter=AverageMeter()
    loss_pseudo_rec_meter=AverageMeter()
    loss_cluster_balance_meter=AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)
    epoch_timer=Timer()

    model = models['model']
    model.eval()

    # generate pseudo labels from person perfeatures with k-means
    if cfg.mode == 'PAC_DC':
        model_pac_net = models['model_pac_net']
        model_pac_net.eval()

        batch_idx = -1
        for batch_data in tqdm(data_loader):
            batch_idx += 1
            with torch.no_grad():
                for key in range(len(batch_data)):
                    if torch.is_tensor(batch_data[key]):
                        batch_data[key] = batch_data[key].to(device=device)

                batch_size=batch_data[0].shape[0]
                num_frames=batch_data[0].shape[1]
                input_data = {}
                input_data['images_in'] = batch_data[0]
                input_data['boxes_in'] = batch_data[1]
                input_data['actions_in'] = batch_data[2]
                input_data['images_person_in'] = batch_data[4]
    
                # obtain person features
                ret_pac_net = model_pac_net(input_data)
                person_features_pac_net = ret_pac_net['person_features']
                person_features_pac_net_dim = person_features_pac_net.shape[-1]
                if batch_idx == 0:
                    person_features_pac_net_all = torch.zeros(len(data_loader), cfg.batch_size, cfg.num_boxes, person_features_pac_net_dim).to(device=device)
                person_features_pac_net_all[batch_idx, :batch_size] = person_features_pac_net.detach().reshape(batch_size, cfg.num_boxes, -1)
        person_features_pac_net_all = person_features_pac_net_all.reshape(len(data_loader)*cfg.batch_size*cfg.num_boxes, -1)

        print('Clustering in processing...')
        kmeans_gpu = KMeansGPU(n_clusters=cfg.num_actions, mode='euclidean', verbose=0)
        pseudo_action_labels = kmeans_gpu.fit_predict(person_features_pac_net_all)
        pseudo_action_labels = pseudo_action_labels.reshape(len(data_loader), cfg.batch_size*cfg.num_boxes)
        pseudo_action_labels_weight = torch.zeros(cfg.num_actions).to(device=device)
        for i in range(cfg.num_actions):
            pseudo_action_labels_weight[i] = torch.sum(torch.eq(pseudo_action_labels, i).float())
        print('Clustering results:', pseudo_action_labels_weight)
        pseudo_action_labels_weight_inv = (1 / pseudo_action_labels_weight)
        pseudo_action_labels_weight_inv = pseudo_action_labels_weight_inv / torch.sum(pseudo_action_labels_weight_inv)
        print('pseudo_action_labels_weight_inv:', pseudo_action_labels_weight_inv)

    with torch.no_grad():
        for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
            for key in range(len(batch_data_test)):
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)

            batch_size=batch_data_test[0].shape[0]
            num_frames=batch_data_test[0].shape[1]

            actions_in=batch_data_test[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data_test[3].reshape((batch_size,num_frames))
                        
            # Predict actions
            actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            activities_in=activities_in[:,0].reshape((batch_size,))

            # forward
            # ret= model((batch_data_test[0], batch_data_test[1], batch_data_test[4]))
            input_data = {}
            input_data['images_in'] = batch_data_test[0]
            input_data['boxes_in'] = batch_data_test[1]
            input_data['images_person_in'] = batch_data_test[4]
            input_data['actions_in'] = batch_data_test[2]

            # define list for various losses
            loss_list = []

            # recognize actions
            if cfg.mode == 'PAC_REC':
                ret_rec_act = models['model_act_recognizer'](input_data)
                rec_act_labels = torch.argmax(ret_rec_act['actions'], dim=1)
                input_data['actions_in'] = rec_act_labels

            # predict pseudo actions
            if cfg.mode == 'PAC_DC':
                ret_pac_net = model_pac_net(input_data)
                action_scores_pac_net = ret_pac_net['pseudo_scores']
                action_scores_pac_net_sm = F.softmax(action_scores_pac_net, dim=-1)
                action_scores_pac_net = action_scores_pac_net.reshape(batch_size*cfg.num_boxes, -1)
                if cfg.use_pseudo_act_gt:
                    actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
                    pseudo_actions_loss = F.cross_entropy(action_scores_pac_net, actions_in, weight=actions_weights)
                else:
                    if cfg.use_pseudo_act_loss_weight:
                        pseudo_actions_loss = F.cross_entropy(action_scores_pac_net, pseudo_action_labels[batch_idx][:batch_size*cfg.num_boxes], weight=pseudo_action_labels_weight_inv)
                    else:
                        pseudo_actions_loss = F.cross_entropy(action_scores_pac_net, pseudo_action_labels[batch_idx][:batch_size*cfg.num_boxes])
                loss_list.append(pseudo_actions_loss)
                loss_pseudo_rec_meter.update(pseudo_actions_loss.item(), batch_size)

                # clustering balance loss
                if cfg.use_cluster_balance_loss:
                    action_scores_pac_net_sm_mean = torch.mean(action_scores_pac_net_sm, dim=0)
                    action_scores_pac_net_sm_mean_expand = action_scores_pac_net_sm_mean.unsqueeze(1).expand(cfg.num_actions, cfg.num_actions)
                    action_scores_pac_net_sm_prior = F.softmax(torch.tensor(cfg.actions_weights, dtype=torch.float32).to(device=device), dim=-1)
                    action_scores_pac_net_sm_prior_expand = action_scores_pac_net_sm_prior.unsqueeze(0).expand(cfg.num_actions, cfg.num_actions)
                    action_scores_cost_matrix = torch.abs(action_scores_pac_net_sm_prior_expand - action_scores_pac_net_sm_mean_expand)
                    action_scores_cost_matrix = torch.zeros(cfg.num_actions, cfg.num_actions).to(device=device)
                    for i in range(cfg.num_actions):
                        for j in range(cfg.num_actions):
                            action_scores_cost_matrix[i][j] = torch.norm(action_scores_pac_net_sm_prior[i] - action_scores_pac_net_sm_mean[j])
                    action_scores_pac_net_sm_prior_sort = torch.zeros(cfg.num_actions).to(device=device)
                    hungarian = Hungarian(action_scores_cost_matrix.detach().cpu().numpy())
                    hungarian.calculate()
                    hungarian_results = hungarian.get_results()
                    for i in range(cfg.num_actions):
                        mean_idx, prior_idx = hungarian_results[i]
                        action_scores_pac_net_sm_prior_sort[mean_idx] = action_scores_pac_net_sm_prior[prior_idx]
                    cluster_balance_loss = F.mse_loss(action_scores_pac_net_sm_mean, action_scores_pac_net_sm_prior_sort) * 10e2
                    loss_list.append(cluster_balance_loss)
                    loss_cluster_balance_meter.update(cluster_balance_loss.item(), batch_size)

            ret= model(input_data)

            # Predict activities
            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                activities_loss = F.cross_entropy(activities_scores,activities_in)
                loss_list.append(activities_loss)
                activities_labels = torch.argmax(activities_scores,dim=1)
                activities_correct = torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
                activities_accuracy = activities_correct.item() / activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])
                activities_conf.add(activities_labels, activities_in)

            if 'actions' in list(ret.keys()):
                actions_scores = ret['actions']
                actions_weights=torch.tensor(cfg.actions_weights).to(device=device)
                if cfg.mode == 'PAC_DC':
                    actions_loss = F.cross_entropy(actions_scores, action_scores_pac_net_sm) * cfg.actions_loss_weight
                else:
                    actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights) * cfg.actions_loss_weight
                loss_list.append(actions_loss)
                actions_labels=torch.argmax(actions_scores,dim=1)
                actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]
                actions_meter.update(actions_accuracy, actions_scores.shape[0])
                actions_conf.add(actions_labels, actions_in)
                loss_act_meter.update(actions_loss.item(), batch_size)

            # Predict features
            if 'recon_features' in list(ret.keys()):
                recon_features = ret['recon_features']
                original_features = ret['original_features']
                recon_loss = F.mse_loss(recon_features, original_features)
                loss_list.append(recon_loss)
                loss_recon_meter.update(recon_loss.item(), batch_size)

            if 'halting' in list(ret.keys()):
                loss_list.append(ret['halting'])

            # Total loss
            total_loss = sum(loss_list)
            loss_meter.update(total_loss.item(), batch_size)

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'loss_pseudo_rec':loss_pseudo_rec_meter.avg,
        'loss_cluster_balance':loss_cluster_balance_meter.avg,
        'loss_act':loss_act_meter.avg,
        'loss_recon':loss_recon_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
    }
    
    return test_info

def train_collective(data_loader, models, device, optimizer, epoch, cfg):
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)

    model = models['model']
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        # batch_data=[b.to(device=device) for b in batch_data]
        for key in range(len(batch_data)):
            if torch.is_tensor(batch_data[key]):
                batch_data[key] = batch_data[key].to(device=device)

        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]
        actions_in = batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
        activities_in = batch_data[3].reshape((batch_size,num_frames))
        bboxes_num = batch_data[5].reshape(batch_size,num_frames)

        # forward
        # ret= model((batch_data[0], batch_data[1], batch_data[4]))
        input_data = {}
        input_data['images_in'] = batch_data[0]
        input_data['boxes_in'] = batch_data[1]
        input_data['images_person_in'] = batch_data[4]
        ret= model(input_data)

        actions_in_nopad=[]
        for b in range(batch_size):
            N = bboxes_num[b][0]
            actions_in_nopad.append(actions_in[b][0][:N])
        actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)

        loss_list = []
        if 'actions' in list(ret.keys()):
            actions_scores = ret['actions'].reshape(batch_size, cfg.num_boxes, -1)
            actions_scores_nopad=[]
            for b in range(batch_size):
                N = bboxes_num[b][0]
                actions_scores_nopad.append(actions_scores[b][:N])
            actions_scores=torch.cat(actions_scores_nopad,dim=0).reshape(-1, cfg.num_actions)
            actions_loss=F.cross_entropy(actions_scores,actions_in)
            loss_list.append(actions_loss)
            actions_labels=torch.argmax(actions_scores,dim=1)
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            actions_conf.add(actions_labels, actions_in)

        if 'recon_features' in list(ret.keys()):
            recon_features = ret['recon_features']
            original_features = ret['original_features']
            recon_features_nopad=[]
            original_features_nopad=[]
            for b in range(batch_size):
                N = bboxes_num[b][0]
                recon_features_nopad.append(recon_features[b][:, :N, :].reshape(-1,))
                original_features_nopad.append(original_features[b][:, :N, :].reshape(-1,))
            recon_features_nopad=torch.cat(recon_features_nopad,dim=0).reshape(-1,)
            original_features_nopad=torch.cat(original_features_nopad,dim=0).reshape(-1,)
            recon_loss = F.mse_loss(recon_features_nopad, original_features_nopad)
            loss_list.append(recon_loss)
        
        # Total loss
        total_loss = sum(loss_list)
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'activities_conf':activities_conf.value(),
        'activities_MPCA':MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
    }  

    return train_info
        
def test_collective(data_loader, models, device, epoch, cfg):
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)

    model = models['model']
    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            # batch_data=[b.to(device=device) for b in batch_data]
            for key in range(len(batch_data)):
                if torch.is_tensor(batch_data[key]):
                    batch_data[key] = batch_data[key].to(device=device)

            batch_size=batch_data[0].shape[0]
            num_frames=batch_data[0].shape[1]
            
            actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data[3].reshape((batch_size,num_frames))
            bboxes_num=batch_data[5].reshape(batch_size,num_frames)

            # forward
            # ret= model((batch_data[0], batch_data[1], batch_data[4]))
            input_data = {}
            input_data['images_in'] = batch_data[0]
            input_data['boxes_in'] = batch_data[1]
            input_data['images_person_in'] = batch_data[4]
            ret= model(input_data)

            actions_in_nopad=[]
            for b in range(batch_size):
                N = bboxes_num[b][0]
                actions_in_nopad.append(actions_in[b][0][:N])
            actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)

            loss_list = []
            if 'actions' in list(ret.keys()):
                actions_scores = ret['actions'].reshape(batch_size, cfg.num_boxes, -1)
                actions_scores_nopad=[]
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    actions_scores_nopad.append(actions_scores[b][:N])
                actions_scores=torch.cat(actions_scores_nopad,dim=0).reshape(-1, cfg.num_actions)
                actions_loss=F.cross_entropy(actions_scores,actions_in)
                loss_list.append(actions_loss)
                actions_labels=torch.argmax(actions_scores,dim=1)
                actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]
                actions_meter.update(actions_accuracy, actions_scores.shape[0])
                actions_conf.add(actions_labels, actions_in)

            if 'recon_features' in list(ret.keys()):
                recon_features = ret['recon_features']
                original_features = ret['original_features']
                recon_features_nopad=[]
                original_features_nopad=[]
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    recon_features_nopad.append(recon_features[b][:, :N, :].reshape(-1,))
                    original_features_nopad.append(original_features[b][:, :N, :].reshape(-1,))
                recon_features_nopad=torch.cat(recon_features_nopad,dim=0).reshape(-1,)
                original_features_nopad=torch.cat(original_features_nopad,dim=0).reshape(-1,)
                recon_loss = F.mse_loss(recon_features_nopad, original_features_nopad)
                loss_list.append(recon_loss)

            # Total loss
            total_loss = sum(loss_list)
            loss_meter.update(total_loss.item(), batch_size)

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
    }

    return test_info