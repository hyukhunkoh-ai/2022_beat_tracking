#https://tutorials.pytorch.kr/intermediate/dist_tuto.html
import os,sys
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from models.self_supervised import Music2VecModel

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
# print("count:",torch.cuda.device_count())


def train(gpu, args):

    ############################################################
    #  Initialize the process and join up with the other processes
    rank = gpu              
    dist.init_process_group(                                   
    	backend='nccl',   # fastest backend among others( Gloo, NCCL 및 MPI의 세 가지 백엔드)                                   
   		init_method='tcp://127.0.0.1:3456',  # tells the process group where to look for some settings                       
    	world_size=args.world_size,                              
    	rank=rank                                               
    )                                                          
    ############################################################

    torch.manual_seed(0)
    model = Music2VecModel()
   
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 256

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    ###############################################################
    # Wrap the model to each gpu
    dist_model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    # dist_model들 간의 통신 연결 - hook 등록
    ###############################################################



    train_dataset = torchvision.datasets.MNIST(root='MNIST_data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    ################################################################
    # assign different slice of data per the process 
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    ################################################################

    train_loader = torch.utils.data.DataLoader(
    	dataset=train_dataset,
       batch_size=batch_size,
    ##############################
       shuffle=False,            #
    ##############################
       num_workers=4,
       pin_memory=True,
    #############################
      sampler=train_sampler)    # 
    #############################

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = dist_model(images)
            loss = criterion(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N') # number of machine
    parser.add_argument('-g', '--gpus', default=1, type=int, # number of gpus
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, # node rank within all nodes
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    #########################################################
    args.world_size = args.gpus * args.nodes                # total number of processes to run
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    # spawn args.gpus processes, each of which runs train(i, args), where i goes from 0 to args.gpus - 1.#
    #########################################################
    



if __name__ == '__main__':
    main()
