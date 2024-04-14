from __future__ import print_function
import argparse
from ast import Str
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 
import shutil
import time
import random
import torch
from torch.autograd import variable
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from models import *
from progress.bar import Bar
from utils import  Logger, AverageMeter, accuracy, mkdir_p, savefig
import sys
import csv
import timm
from pytorch_lightning import seed_everything




# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Scene Classification Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=64, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=128, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[60],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
#Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ')
parser.add_argument('--archname',  default='resnet18',
                    help='model architecture: ')
parser.add_argument('--manualSeed',default='42', type=int, help='manual seed')
parser.add_argument( '--accpath', default='1', type=str)
parser.add_argument( '--pklpath', default='1', type=str)
#Device options
parser.add_argument('--gpu_id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--traing_ratio', default='0.5', type=str,
                    help='traing_ratio')
parser.add_argument('--op', default='sgd', type=str)
parser.add_argument('--runs', default='runs', type=str)
parser.add_argument('-n', '--num', default='1', type=int)
parser.add_argument('--mode', default='SAG', type=str)
parser.add_argument('--dataset', default='airound', type=str,
                    help='dataset')
parser.add_argument('--dim', default='128', type=int)
parser.add_argument('--dp', default='0.4', type=float)
parser.add_argument('--at', default='1', type=int)
parser.add_argument('--fusion', default='1', type=int)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

args.runs=args.runs
#dataset
args.data='E:/airound/aerial-'+str(args.traing_ratio)+'-'+str(args.num)


num_class=11

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
seed = args.manualSeed
seed_everything(seed)

best_acc = 0  # best test accuracy
best_epoch=0
arch=args.archname

if args.mode=="SAG":
    from torchvision.datasets import SAG_ImageFolder as imagefolder
elif args.mode=="SA":
    from torchvision.datasets import SA_ImageFolder as imagefolder
elif args.mode=="SG":
    from torchvision.datasets import SG_ImageFolder as imagefolder
elif args.mode=="AG":
    from torchvision.datasets import AG_ImageFolder as imagefolder
elif args.mode=="A":
    from torchvision.datasets import A_ImageFolder as imagefolder
elif args.mode=="G":
    from torchvision.datasets import G_ImageFolder as imagefolder
elif args.mode=="S":
    from torchvision.datasets import S_ImageFolder as imagefolder
def main():
    
    begintime=time.time()
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    a_normalize=transforms.Normalize(mean=[0.401, 0.415, 0.381],
                                     std=[0.231,  0.207, 0.210])
    g_normalize=transforms.Normalize(mean=[0.474, 0.489, 0.464],
                                     std=[0.236,   0.234,  0.271])
    train_loader = torch.utils.data.DataLoader(
        imagefolder(traindir, transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            a_normalize,
        ]),transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            g_normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True,persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        imagefolder(valdir, transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            a_normalize,
        ]),transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            g_normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=2, pin_memory=True,persistent_workers=True)

    # create model
    pklpath_s=arch+"-"+"S"+"-"+str(args.traing_ratio)+"-"+str(args.num)
    if args.arch=="resnet":      
        model=resnet(num_class,args.archname,args.mode,args.at,args.fusion,args.dim,args.dp)
            


    model.cuda()
    cudnn.benchmark = True 
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    size=sum(p.numel() for p in model.parameters())/1000000.0
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                 milestones=args.schedule,
    #                                                 gamma=args.gamma)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=10,factor=0.1,min_lr=0.0001,eps=0.0003)



    # log
    title = 'RSNet-' + args.arch
    pklpath=arch+"-"+args.mode+"-"+str(args.at)+"-"+str(args.fusion)+"-"+str(args.dim)+"-"+str(args.dp)+"-"+str(args.traing_ratio)+"-"+str(args.num)
    check_path(os.path.join('results',args.accpath))
    logger = Logger( os.path.join('results',args.accpath, pklpath+'.txt'), title=title)
    logger.set_names(['Epoch','Learning Rate1', 'Train Loss', 'Train Acc.'])
    if os.path.exists(os.path.join('results',args.accpath, args.runs+".csv")):
        f = open(os.path.join('results',args.accpath, args.runs+".csv"),'a',encoding='utf-8',newline='')
        csv_writer = csv.writer(f)
    else:
        f = open(os.path.join('results',args.accpath, args.runs+".csv"),'w',encoding='utf-8',newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(['arch','acc','dataset','trainratio','op','lr','schedule','bz','epoch','num','size','run_time','at','fusion','dim','dp'])


    # Train and val
    acc=[]
    t=0
    for epoch in range(start_epoch, 500): 
        t=epoch
        if optimizer.param_groups[0]['lr']<0.001:
            break
        print('\nEpoch: [%d | %d] LR1: %f ' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)  
        logger.append([epoch + 1,optimizer.param_groups[0]['lr'], train_loss,  train_acc])          
        lr_scheduler.step(train_loss)
    train_loss, test_acc,test1,test2,test3 = test(val_loader, model, criterion, 1, use_cuda)
    acc.append(test_acc)
    endtime=time.time()
    runtime=endtime-begintime
    csv_writer.writerow([arch+"-"+args.mode, '%.4f'%test_acc,
                        args.dataset,args.traing_ratio,args.op,
                        str(args.lr),str(args.schedule),str(args.train_batch),
                        str(t),str(args.num),'%.2f'%size,str(runtime),str( args.at),str( args.fusion),str( args.dim),str( args.dp),'%.4f'%test1,'%.4f'%test2,'%.4f'%test3])
    f.close()
    check_path(os.path.join('E:\\results\\airound', args.pklpath))
    filepath = os.path.join('E:\\results\\airound', args.pklpath, pklpath+".pkl")
    torch.save(model.state_dict(), filepath)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (s_sample,a_sample,g_sample,targets,path) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            s_sample = torch.Tensor(s_sample).cuda()
            a_sample = a_sample.cuda()
            g_sample = g_sample.cuda()
            targets = targets.cuda()
        s_sample,a_sample,g_sample, targets = torch.autograd.Variable(s_sample), torch.autograd.Variable(a_sample),torch.autograd.Variable(g_sample),torch.autograd.Variable(targets)
        outputs = model(s_sample,a_sample,g_sample)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), s_sample.size(0))
        top1.update(prec1.item(), s_sample.size(0))
        top5.update(prec5.item(), s_sample.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    topa = AverageMeter()
    topg = AverageMeter()
    topag = AverageMeter()

    # switch to evaluate mode
    model.eval()
    torch.cuda.empty_cache()
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    with torch.no_grad():
        for batch_idx, (s_sample,a_sample,g_sample,targets,path) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            torch.cuda.empty_cache()
            if use_cuda:
                s_sample = torch.Tensor(s_sample).cuda()
                a_sample = a_sample.cuda()
                g_sample = g_sample.cuda()
                targets = targets.cuda()
            s_sample,a_sample,g_sample, targets = torch.autograd.Variable(s_sample), torch.autograd.Variable(a_sample),torch.autograd.Variable(g_sample),torch.autograd.Variable(targets)
            outputs= model(s_sample,a_sample,g_sample)
            bs = s_sample.size(0)
            zero_a_sample=torch.zeros(bs,3,224,224).cuda()
            zero_g_sample=torch.zeros(bs,3,224,224).cuda()
            zero_s_sample=torch.zeros(bs,13,224,224).cuda()
            loss = criterion(outputs, targets)
            a_outputs=model(s_sample,zero_a_sample,g_sample)
            g_outputs=model(s_sample,a_sample,zero_g_sample)
            ag_outputs=model(zero_s_sample,a_sample,g_sample)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            prec_a, _ = accuracy(a_outputs.data, targets.data, topk=(1, 5))
            prec_g, _ = accuracy(g_outputs.data, targets.data, topk=(1, 5))
            prec_ag, _ = accuracy(ag_outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), s_sample.size(0))
            top1.update(prec1.item(), s_sample.size(0))
            topa.update(prec_a.item(), s_sample.size(0))
            topg.update(prec_g.item(), s_sample.size(0))
            topag.update(prec_ag.item(), s_sample.size(0))
            top5.update(prec5.item(), s_sample.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | topa: {topa: .4f} | topg: {topg: .4f} | topag: {topag: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        topa=topa.avg,
                        topg=topg.avg,
                        topag=topag.avg,
                        )
            bar.next()
    bar.finish()
    return (losses.avg, top1.avg, topag.avg, topa.avg, topg.avg)

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def check_path(path):
    # print(path)
    _path = path
    if not os.path.exists(_path):   	
        os.makedirs(_path, exist_ok=True)
if __name__ == '__main__':
    main()

