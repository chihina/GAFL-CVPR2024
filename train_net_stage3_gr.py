'''
    Fine-tune the model with group activity labels.
'''

import torch
import torch.optim as optim

import time
import random
import os
import sys
import wandb
from collections import OrderedDict

from config import *
from volleyball import *
from collective import *
from dataset import *
from infer_model import *
from infer_model_prev import *
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
    gcnnet_list={'group_relation_volleyball':GroupRelation_volleyball,
                 'group_relation_collective':GroupRelation_volleyball,
                 }
    
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
            state = torch.load(cfg.stage2model)
            model.load_state_dict(state['state_dict'])
            print_log(cfg.log_path, 'Loading stage2 model: ' + cfg.stage2model)
        else:
            print_log(cfg.log_path, 'Not loading stage1 or stage2 model.')
    elif cfg.training_stage==3:
        GCNnet = gcnnet_list[cfg.inference_module_name]
        model = GCNnet(cfg)
        state_dict = torch.load(cfg.stage2model)['state_dict']
        new_state_dict=OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
        
        if cfg.load_backbone_stage3:
            model.load_state_dict(new_state_dict, strict=False)
            print_log(cfg.log_path, 'Loading stage2 model for stage3: ' + cfg.stage2model)
        else:
            if cfg.use_jae_loss:
                model.loadmodel(cfg.stage1_model_path)
    else:
        assert(False)
    
    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)

    model=model.to(device=device)
    
    model.train()
    if cfg.set_bn_eval:
        model.apply(set_bn_eval)
    
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.train_learning_rate,weight_decay=cfg.weight_decay)

    train_list={'volleyball':train_volleyball, 'collective':train_collective}
    test_list={'volleyball':test_volleyball, 'collective':test_collective}
    train=train_list[cfg.dataset_name]
    test=test_list[cfg.dataset_name]
    
    if cfg.test_before_train:
        test_info=test(validation_loader, model, device, 0, cfg)
        print(test_info)
    
    # Training iteration
    best_result = {'epoch':0, 'activities_acc':0, 'actions_acc':0, 'loss':100000000000000}
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
            
        # One epoch of forward and backward
        train_info=train(training_loader, model, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)
        for wandb_loss_name in cfg.wandb_loss_list:
            wandb.log({f"Train {wandb_loss_name}": train_info[wandb_loss_name]}, step=epoch)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info=test(validation_loader, model, device, epoch, cfg)
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
                torch.save(state, cfg.result_path+'/best_model.pth')
                print('Best model saved.')
            print_log(cfg.log_path, 
                    #   'Best group activity accuracy: %.2f%% at epoch #%d.'%(best_result['activities_acc'], best_result['epoch']))
                    #   'Best individual action accuracy: %.2f%% at epoch #%d.'%(best_result['actions_acc'], best_result['epoch']))
                      'Best loss: %.2f%% at epoch #%d.'%(best_result['loss'], best_result['epoch']))

            # Save model
            if cfg.training_stage in [2, 3]:
                # None
                # if test_info['activities_acc'] > 93.1:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
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
   
def train_volleyball(data_loader, model, device, optimizer, epoch, cfg):
    train_with_action = False
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)
    epoch_timer=Timer()

    for batch_idx, batch_data in enumerate(data_loader):

        if batch_idx % 100 == 0 and batch_idx > 0:
        # if batch_idx % 850 == 0 and batch_idx > 0:
            print('Training in processing {}/{}, Loss: {:.4f}'.format(batch_idx, len(data_loader), loss_meter.avg))

        model.train()
        if cfg.set_bn_eval:
            model.apply(set_bn_eval)
    
        # prepare batch data
        # batch_data=[b.to(device = device) for b in batch_data]
        for key in range(len(batch_data)):
            if torch.is_tensor(batch_data[key]):
                batch_data[key] = batch_data[key].to(device=device)

        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
        activities_in=batch_data[3].reshape((batch_size,num_frames))

        actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
        activities_in=activities_in[:,0].reshape((batch_size,))

        # actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))

        # forward
        # ret= model((batch_data[0], batch_data[1], batch_data[4]))
        input_data = {}
        input_data['images_in'] = batch_data[0]
        input_data['boxes_in'] = batch_data[1]
        input_data['images_person_in'] = batch_data[4]
        input_data['actions_in'] = batch_data[2]
        ret= model(input_data)

        loss_list = []

        # Predict joint attention
        if 'estimated_ja' in list(ret.keys()):
            estimated_ja = ret['estimated_ja']
            gt_ja = batch_data[6].reshape((batch_size, num_frames, 4))
            gt_ja_center = (gt_ja[:, :, 0:2] + gt_ja[:, :, 2:4]) / 2
            ja_loss = F.mse_loss(estimated_ja, gt_ja_center)
            loss_list.append(ja_loss)

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
            actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights) * cfg.actions_loss_weight
            loss_list.append(actions_loss)
            actions_labels = torch.argmax(actions_scores, dim=1)
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            actions_conf.add(actions_labels, actions_in)

        # Predict features
        if 'recon_features' in list(ret.keys()):
            recon_features = ret['recon_features']
            original_features = ret['original_features']
            recon_loss = F.mse_loss(recon_features, original_features)
            loss_list.append(recon_loss)

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
        'activities_acc':activities_meter.avg*100,
        'activities_conf':activities_conf.value(),
        'activities_MPCA':MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
    }      

    return train_info
        
    
def test_volleyball(data_loader, model, device, epoch, cfg):
    model.eval()
    train_with_action = False
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)
    epoch_timer=Timer()

    with torch.no_grad():
        for batch_data_test in data_loader:
            # prepare batch data
            # batch_data_test=[b.to(device=device) for b in batch_data_test]
            for key in range(len(batch_data_test)):
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)

            batch_size=batch_data_test[0].shape[0]
            num_frames=batch_data_test[0].shape[1]

            actions_in=batch_data_test[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data_test[3].reshape((batch_size,num_frames))
            
            # forward
            # actions_scores,activities_scores=model((batch_data_test[0],batch_data_test[1]))
            # activities_scores = model((batch_data_test[0], batch_data_test[1]))
            # ret = model((batch_data_test[0], batch_data_test[1]))
            # ret= model((batch_data_test[0], batch_data_test[1], batch_data_test[4]))
            input_data = {}
            input_data['images_in'] = batch_data_test[0]
            input_data['boxes_in'] = batch_data_test[1]
            input_data['images_person_in'] = batch_data_test[4]
            input_data['actions_in'] = batch_data_test[2]
            ret= model(input_data)

            # Predict actions
            actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            activities_in=activities_in[:,0].reshape((batch_size,))

            loss_list = []

            # Predict joint attention
            if 'estimated_ja' in list(ret.keys()):
                estimated_ja = ret['estimated_ja']
                gt_ja = batch_data_test[6].reshape((batch_size, num_frames, 4))
                gt_ja_center = (gt_ja[:, :, 0:2] + gt_ja[:, :, 2:4]) / 2
                ja_loss = F.mse_loss(estimated_ja, gt_ja_center)
                loss_list.append(ja_loss)

            # Predict activities
            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                activities_loss = F.cross_entropy(activities_scores,activities_in)
                loss_list.append(activities_loss)
                activities_labels = torch.argmax(activities_scores,dim=1)
                # Save wrong samples
                # if torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float()) == 0:
                #     wrong.append(flag)
                # if flag == 1336: # 1336
                #     np.savetxt('vis/wrong_samples.txt', wrong)
                # flag += 1

                activities_correct = torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
                activities_accuracy = activities_correct.item() / activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])
                activities_conf.add(activities_labels, activities_in)

            if 'actions' in list(ret.keys()):
                actions_scores = ret['actions']
                actions_weights=torch.tensor(cfg.actions_weights).to(device=device)
                actions_loss=F.cross_entropy(actions_scores,actions_in,weight=actions_weights)
                loss_list.append(actions_loss)
                actions_labels=torch.argmax(actions_scores,dim=1)
                actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]
                actions_meter.update(actions_accuracy, actions_scores.shape[0])
                actions_conf.add(actions_labels, actions_in)

            # Predict features
            if 'recon_features' in list(ret.keys()):
                recon_features = ret['recon_features']
                original_features = ret['original_features']
                recon_loss = F.mse_loss(recon_features, original_features)
                loss_list.append(recon_loss)

            if 'halting' in list(ret.keys()):
                loss_list.append(ret['halting'])

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


def train_collective(data_loader, model, device, optimizer, epoch, cfg):
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    actions_conf = ConfusionMeter(cfg.num_actions)
    activities_conf = ConfusionMeter(cfg.num_activities)
    epoch_timer=Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
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

        # forward
        # actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
        # activities_scores = model((batch_data[0], batch_data[1], batch_data[4]))
        input_data = {}
        input_data['images_in'] = batch_data[0]
        input_data['boxes_in'] = batch_data[1]
        input_data['images_person_in'] = batch_data[4]
        ret= model(input_data)

        activities_in = batch_data[3].reshape((batch_size,num_frames))
        bboxes_num = batch_data[5].reshape(batch_size,num_frames)

        actions_in = batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
        actions_in_nopad=[]
        if cfg.training_stage==1:
            # actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
            bboxes_num = bboxes_num.reshape(batch_size*num_frames,)
            for bt in range(batch_size*num_frames):
                N=bboxes_num[bt]
                actions_in_nopad.append(actions_in[bt,:N])
        else:
            for b in range(batch_size):
                N = bboxes_num[b][0]
                actions_in_nopad.append(actions_in[b][0][:N])
        actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
        if cfg.training_stage==1:
            activities_in = activities_in.reshape(-1,)
        else:
            activities_in = activities_in[:,0].reshape(batch_size,)

        loss_list = []
        # Predict actions
        if 'actions' in list(ret.keys()):
            actions_scores = ret['actions'].reshape(batch_size, cfg.num_boxes, -1)
            actions_scores_nopad=[]
            for b in range(batch_size):
                N = bboxes_num[b][0]
                actions_scores_nopad.append(actions_scores[b][:N])
            actions_scores=torch.cat(actions_scores_nopad,dim=0).reshape(-1, cfg.num_actions)
            actions_loss=F.cross_entropy(actions_scores,actions_in,weight=None)
            loss_list.append(actions_loss)
            actions_labels=torch.argmax(actions_scores,dim=1)  #B*T*N,
            actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            actions_conf.add(actions_labels, actions_in)

        # Predict activities
        if 'activities' in list(ret.keys()):
            activities_scores = ret['activities']
            activities_loss=F.cross_entropy(activities_scores,activities_in)
            loss_list.append(activities_loss)
            activities_labels=torch.argmax(activities_scores,dim=1)  #B*T,
            activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
            activities_accuracy=activities_correct.item()/activities_scores.shape[0]
            activities_meter.update(activities_accuracy, activities_scores.shape[0])
            activities_conf.add(activities_labels, activities_in)

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
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
    }

    return train_info
        
def test_collective(data_loader, model, device, epoch, cfg):
    model.eval()
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    actions_conf = ConfusionMeter(cfg.num_actions)
    activities_conf = ConfusionMeter(cfg.num_activities)
    epoch_timer=Timer()
    # flag = 0
    # wrong = []
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
            input_data = {}
            input_data['images_in'] = batch_data[0]
            input_data['boxes_in'] = batch_data[1]
            input_data['images_person_in'] = batch_data[4]
            ret= model(input_data)

            actions_in_nopad=[]
            if cfg.training_stage==1:
                actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
                bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
                for bt in range(batch_size*num_frames):
                    N=bboxes_num[bt]
                    actions_in_nopad.append(actions_in[bt,:N])
            else:
                for b in range(batch_size):
                    N=bboxes_num[b][0]
                    actions_in_nopad.append(actions_in[b][0][:N])
            actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
            if cfg.training_stage==1:
                activities_in=activities_in.reshape(-1,)
            else:
                activities_in=activities_in[:,0].reshape(batch_size,)

            loss_list = []
            # Predict actions
            if 'actions' in list(ret.keys()):
                actions_scores = ret['actions'].reshape(batch_size, cfg.num_boxes, -1)
                actions_scores_nopad=[]
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    actions_scores_nopad.append(actions_scores[b][:N])
                actions_scores=torch.cat(actions_scores_nopad,dim=0).reshape(-1, cfg.num_actions)
                actions_loss=F.cross_entropy(actions_scores,actions_in,weight=None)
                loss_list.append(actions_loss)
                actions_labels=torch.argmax(actions_scores,dim=1)  #B*T*N,
                actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]
                actions_meter.update(actions_accuracy, actions_scores.shape[0])
                actions_conf.add(actions_labels, actions_in)

            # Predict activities
            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                activities_loss=F.cross_entropy(activities_scores,activities_in)
                loss_list.append(activities_loss)
                activities_labels=torch.argmax(activities_scores,dim=1)  #B*T,
                activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
                activities_accuracy=activities_correct.item()/activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])
                activities_conf.add(activities_labels, activities_in)

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