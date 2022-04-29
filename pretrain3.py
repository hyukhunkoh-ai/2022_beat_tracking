# from torch._C import T
from torch.autograd.grad_mode import no_grad
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch import nn
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import datetime
from dataset import SelfSupervisedDataset
from models.self_supervised import Music2VecModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

#학습
when_to_load = {'0':True,'1':True,'2':True,'3':True}

def pretrain(gpu, args, pretrain_dataset_list, num_epochs, bs, steps_per_epoch):
    
    
    ############################################################
    #  Initialize the process and join up with the other processes
    rank = gpu              
    dist.init_process_group(                                   
    	backend='nccl',   # fastest backend among others( Gloo, NCCL 및 MPI의 세 가지 백엔드)                                   
   		init_method='tcp://127.0.0.1:5678',  # tells the process group where to look for some settings                       
    	world_size=args.gpus,                              
    	rank=rank                                               
    )                                                          
    ############################################################
    torch.cuda.set_device(gpu) 
  
    batch_size = 30
    
    model = Music2VecModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0005,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.08
    )

    model.cuda(gpu)
    ###############################################################
    # Wrap the model to each gpu
    dist_model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    if os.path.isfile('./finetune_model_52.pth'):
        dist.barrier()
        if when_to_load[str(gpu)]:
            print('load successfully')
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}    
            dist_model.module.load_state_dict(torch.load('./finetune_model_52.pth', map_location=map_location))
            when_to_load[str(gpu)] = False

    ################################################################
    # assign different slice of data per the process 
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	pretrain_dataset_list,
    	num_replicas=args.gpus,
    	rank=rank
    )
    ################################################################
    
    train_loader = torch.utils.data.DataLoader(
        dataset=pretrain_dataset_list,
        batch_size=bs,
    ##############################
        shuffle=False,            #
    ##############################
        pin_memory=True,
    #############################
        sampler=train_sampler)    # 

    start = datetime.datetime.now()
    total_step = len(train_loader)
    total_loss = []
    best_epoch = 0
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        total_trloss = 0
        for i, batch in enumerate(train_loader):
            inputs, attention_masks = batch

            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)

            model.to(device)
            optimizer.zero_grad()
            loss = model.calculate_loss(inputs, attention_masks)
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            if gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    num_epochs, 
                    i + 1, 
                    total_step,
                    loss*10000)
                    #np.mean(total_loss))
                )

        if gpu == 0:
            print(np.mean(total_loss))
            print(best_epoch)
            if np.mean(total_loss) < best_trloss:
                print(epoch)
                print('------------save & loss below---------------')
                best_epoch = epoch
                best_trloss = np.mean(total_loss)
                torch.save(dist_model.module.state_dict(), f'music2vec_pretrain_{best_epoch}.pth')

    if gpu == 0:
        print("Training complete in: " + str(datetime.datetime.now() - start))
        torch.save(dist_model.module.state_dict(), 'music2vec_pretrain.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='how to train?')
    parser.add_argument('--unlabel_dir', type=str, default='/beat_tracking/unlabel')
    parser.add_argument('--gpus', default=4,type=int,help='how many gpu use?')
    parser.add_argument('--devices', help='what gpu use?',default='0,1,2,3')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']= args.devices

    # memoization을 하게끔 만들었음에도 fma_30가 엄청 크기 때문에 데이터로딩이 오래걸리므로. 테스트 시 fma_30 dataset을 생략하는 것이 유리함
    dataset_types = ["60_excerpts_30", "extended_ballroom_30", "acm_mirum_tempo_30_60", "fma_30", "openmic_10"]
    #dataset_types = ["60_excerpts_30", "extended_ballroom_30", "acm_mirum_tempo_30_60", "openmic_10"]

    pretrain_datasets = []
    num_files = 0

    for dataset_type in dataset_types:
        print("Now loading:", dataset_type)
        audio_dir = os.path.join(args.unlabel_dir, dataset_type)
        dataset = SelfSupervisedDataset(audio_dir)

        pretrain_datasets.append(dataset)
        num_files += len(dataset)

    pretrain_dataset_list = torch.utils.data.ConcatDataset(pretrain_datasets)

    num_epochs = 4
    bs = 2
    steps_per_epoch = num_files // bs + 1

    mp.spawn(pretrain, nprocs=args.gpus,args=(args,pretrain_dataset_list,num_epochs,bs,steps_per_epoch))
