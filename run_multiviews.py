'''
Multiview for MVSPCL
Created by Liyan 2023.12.15
'''
import pandas as pd
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils import *
import numpy as np
import os
import sys
###mv+proto##！
from loader import *
from models_proto import *
######
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



parser = argparse.ArgumentParser(description='MVSPCL in PyTorch')
parser.add_argument('--gpu', default=0, type=int, help='GPU device idx to use.')
parser.add_argument('--outf', default='./MVSPCL/demo/log/')
parser.add_argument( "-p","--print_freq", default=10,type=int,metavar="N", help="print frequency (default: 10)")

 
parser.add_argument('--data', default='0', type=int,
                    help='choice of dataset, 0-Scene15')
parser.add_argument('--view_num', type=int, default=3, help='the number of views')
parser.add_argument( "-beta","--beta", default=1,type=int, help="Ablation:trade-off parameter for CL_proto_loss (default= IR)")

parser.add_argument('-bs', '--batch_size', default=512, type=int, help='number of batch size,default=256')
parser.add_argument('-e', '--epochs', default=100, type=int, help='number of epochs to run')
parser.add_argument('-lr', '--learn_rate', default=1e-3, type=float, help='learning rate of adam,default=0.001')
parser.add_argument('--fea_out', type=int, default=128, help='the output dimension of the last linear layer')

### CL loss 
parser.add_argument( "-c","--C", default=0.2,type=int, help="penalty parameter for CL_loss (default: 0.2)")
parser.add_argument('-m', '--proto_m', default=0.99, type=float, help='momentum for computing the momving average of prototypes ,default=0.99')
parser.add_argument('-t', '--temperature', default=0.09, type=float, help='temperature t in Softmax function 0.09')  

parser.add_argument('--gamma', type=float, default=3, help='the power of the weight for each view')

parser.add_argument('--reweighted', default=True, type=bool, help='reweighted or not')
parser.add_argument('--type', default=2, type=int, help='type to run:0-MVSCL,1-MVPCL,2-MVSPCL')


def main():
    args = parser.parse_args()
    data_name = ['Scene15']
   
    if not os.path.exists(args.outf):
        os.mkdir(args.outf)
    folder_name = args.outf+data_name[args.data]+'_'+str(args.type)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logname = 'gamma='+str(args.gamma)+'_t='+str(args.temperature)+'_m='+str(args.proto_m)+'.txt'
    log_file_path = os.path.join(folder_name, logname)
    fh = logging.FileHandler(log_file_path)     
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    NetSeed = 64
    np.random.seed(NetSeed)
    torch.backends.cudnn.deterministic = True 
    torch.manual_seed(NetSeed)  
    torch.cuda.manual_seed(NetSeed)  
    

    train_loader, test_loader, divide_seed,class_num,dim_list,test_data,test_label = \
                    loader(args.batch_size,data_name[args.data],args.view_num)
    
    model = PICL(dim_list,class_num,args).to(args.gpu)


    criterion = ContrastiveLoss().to(args.gpu)
    criterion_CE = nn.CrossEntropyLoss().to(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)



    logging.info("args:%s",args)
    logging.info("model:%s",model)
    logging.info(
        "=============== Training begin, batch_size = {}, NetSeed = {}, DivSeed = {}============".format( args.batch_size, NetSeed,
            divide_seed))

    # 5. train
    ################################### train #################################
    top1_list,top5_list= [], []
    losses_train, losses_val =[],[]
    best_acc1 = 0
    best_acc5= 0
    best_prec,best_recall,best_f1=0,0,0
    weight_var = torch.ones(args.view_num) * (1 / args.view_num)#6
    weight_var = weight_var.to("cuda")
    start_time = time.time()

    for epoch in range(0, args.epochs + 1):
        logging.info('*********************** Training begin epoch {}***************************'.format(epoch))
        if epoch == 0:                 #debug
            with torch.no_grad():
                losses,acc1,weight_var= train(train_loader, model, criterion,criterion_CE, optimizer, epoch,class_num,weight_var, args)
        else:
            adjust_learning_rate(optimizer, epoch,args)
            losses,acc1,weight_var= train(train_loader,model, criterion,criterion_CE, optimizer, epoch,class_num,weight_var,args)
        logging.info('=========================== Testing in the validation set =======================')
        acc1,acc5,prec,recall,f1,loss_val = test(test_loader, model,epoch, criterion,criterion_CE,weight_var,class_num, args)
        # remember best acc@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5,best_acc5)
        best_prec = max(prec, best_prec)
        best_recall = max(recall, best_recall)
        best_f1= max(f1,best_f1)

        # # # save the checkpoint
        # filename = os.path.join(folder_name, 'epoch_%d.pth.tar' %epoch)
        # if is_best:
        #     save_checkpoint(
        #     {
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'best_acc5':best_acc5,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best,folder_name, filename)



        # draw the picture!!
        acc1,acc5 = acc1.cpu().numpy(),acc5.cpu().numpy()
        top1_list.append(acc1)
        top5_list.append(acc5)

        losses_train.append(losses)
        losses_val.append(loss_val)

    best_acc1= best_acc1.detach().cpu().numpy()
    best_acc5 = best_acc5.detach().cpu().numpy()
    logging.info("Best classification: acc1={:.4f},pre={:.4f},recall={:.4f},f1={:.4f},acc5={:.4f}".format(best_acc1,best_prec,best_recall,best_f1,best_acc5))
    end_time = time.time()
    spend_time = round(end_time - start_time, 2)
    logging.info("Total time = {} s".format(spend_time))
    logging.info('!!!!!!!!!!!!DONE!!!!!!!!!!!!!')




# 定义CL loss
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, features,mask,proto,proto_mask, args):
        '''
        InfoNCE+protoNCE
        '''
        bs = args.batch_size
        device = args.gpu
        mask = mask.float().detach().to(device)
        proto_mask = proto_mask.float().detach().to(device)
        q = features[:bs]
        k = features[bs:]
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        sim_qk = torch.div(torch.matmul(q, k.T),args.temperature)  # bs*bs
        sim_qp = torch.div(torch.matmul(q, proto.T),args.temperature)  # bs*class_num

        
        logits_max, _ = torch.max(sim_qk, dim=1, keepdim=True)
        logits = sim_qk - logits_max.detach()
        logits_max_proto, _ = torch.max(sim_qp, dim=1, keepdim=True)
        logit_proto = sim_qp - logits_max_proto.detach()

        mask_pos = torch.zeros_like(mask)
        mask_neg = torch.zeros_like(mask)
        mask_pos[mask == 1] = 1 
        mask_neg[mask == 0] = 1   

       
        exp_logits_pos = torch.exp(logits) * mask_pos
        exp_logits_neg = torch.exp(logits) * mask_neg
        

        numerator = exp_logits_pos
        denominator=(exp_logits_pos).sum(1, keepdim=True) + (exp_logits_neg).sum(1, keepdim=True)
        log_prob = torch.log(numerator/denominator+1e-12) # bs*bs
        log_prob_proto = logit_proto - torch.log(torch.exp(logit_proto).sum(1, keepdim=True)) 
        
        loss_ins = -((mask_pos * log_prob).sum(1))/ mask_pos.sum(1) 
        loss_proto= -(proto_mask * log_prob_proto).sum(1) 
        if args.type == 0:  # SCL
            args.beta = 0   # m
            loss_CL = (loss_ins).mean()
        elif args.type ==1: # PCL
            args.beta = 1   
            loss_CL = (loss_proto).mean()
        elif args.type ==2: # SPCL
            loss_CL = (loss_ins+args.beta*loss_proto).mean()
        return loss_CL

# train
def train(train_loader, model, criterion,criterion_CE, optimizer, epoch,class_num,weight_var, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e") 
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )
    
    model.train()
    time0 = time.time()

    for batch_idx, (x_list,labels) in enumerate(train_loader):  # x0:1024*1*20  x1:1024*1*59
        proto_labels = torch.arange(0, class_num).to(args.gpu)
        mask= get_real_labels(labels,labels)  
        mask=  mask.to(args.gpu) 
        proto_mask = get_real_labels(labels,proto_labels)
        
        fea, logit,prototypes= model(x_list,labels,args) 
        loss = torch.zeros(1).to("cuda")

        ###### adptive weight ###
        weight_up_list = []
        for v in range(args.view_num):    
            
            loss_CL = criterion(fea[v],mask,prototypes[v],proto_mask,args)  
            loss_CE = criterion_CE(logit[v],labels) 
            loss_temp = args.C*loss_CL+loss_CE
            loss += (weight_var[v] ** args.gamma) * loss_temp      

            weight_up_temp = loss_temp ** (1 / (1 - args.gamma))
            weight_up_list.append(weight_up_temp)

        output_var = torch.stack(logit)

        weight_var = weight_var.unsqueeze(1)
        weight_var = weight_var.unsqueeze(2)
        weight_var = weight_var.expand(weight_var.size(0), args.batch_size, class_num)#v*bs*class_num
        output_weighted = weight_var * output_var
        output_weighted = torch.sum(output_weighted, 0)  # bs*class_num

        weight_var = weight_var[:, :, 1]
        weight_var = weight_var[:, 1]
        weight_up_var = torch.FloatTensor(weight_up_list).to("cuda") 
        weight_down_var = torch.sum(weight_up_var)
        weight_var = torch.div(weight_up_var, weight_down_var)
    
       
        acc1, acc5 = accuracy(output_weighted, labels, topk=(1, 5))
        losses.update(loss.item(), args.batch_size)
        top1.update(acc1[0], args.batch_size)
        top5.update(acc5[0], args.batch_size)
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_time.update(time.time() - time0)
        if batch_idx % args.print_freq == 0:
            progress_info=progress.display(batch_idx)

 

    if epoch % 1 == 0:
        logging.info("loss. = {}, acc1. = {}, acc5. = {}".format(losses.avg,top1.avg,top5.avg))
        
    return losses.avg,top1.avg,weight_var   



def test(test_loader, model,epoch, criterion,criterion_CE,weight_var,class_num, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e") 
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    prec = AverageMeter("prec",":6.2f")
    recall = AverageMeter("recall",":6.2f")
    f1 = AverageMeter("f1",":6.2f")

    time0 = time.time()
    progress = ProgressMeter(
        len(test_loader), [batch_time,losses, top1, top5,prec,recall,f1], prefix="Test: "
    )
    
    model.eval()   

    with torch.no_grad():
        for batch_idx, (x_list,labels) in enumerate(test_loader):  # x0:1024*1*20  x1:1024*1*59
            proto_labels = torch.arange(0, class_num).to(args.gpu)
            mask= get_real_labels(labels,labels)   
            mask=  mask.to(args.gpu)  
            proto_mask = get_real_labels(labels,proto_labels)  
            
            fea, logit,prototypes= model(x_list,labels,args) 

            loss = torch.zeros(1).to("cuda")
            ###### test #####
            for v in range(args.view_num):
                loss_CL = criterion(fea[v],mask,prototypes[v],proto_mask,args)   
                loss_CE = criterion_CE(logit[v],labels) 
                loss_temp = args.C*loss_CL+loss_CE
                loss += (weight_var[v] ** args.gamma) * loss_temp      
            output_var = torch.stack(logit)

            weight_var = weight_var.unsqueeze(1)
            weight_var = weight_var.unsqueeze(2)
            weight_var = weight_var.expand(weight_var.size(0), args.batch_size, class_num)#v*bs*class_num
            output_weighted = weight_var * output_var
            output_weighted = torch.sum(output_weighted, 0)  # bs*class_num

            weight_var = weight_var[:, :, 1]
            weight_var = weight_var[:, 1]
            
            

            # 计算最终分类结果
            acc1, acc5 = accuracy(output_weighted, labels, topk=(1, 5))
            met = classification_metric(output_weighted,labels,average='macro', decimals=4)
            losses.update(loss.item(), labels.size(0))
            top1.update(acc1[0], labels.size(0))
            top5.update(acc5[0], labels.size(0))
            prec.update(met['precision'],labels.size(0))
            recall.update(met['recall'], labels.size(0))
            f1.update(met['f_score'], labels.size(0))
    

            # measure elapsed time
            batch_time.update(time.time() - time0)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                progress_info = progress.display(batch_idx)
        
    return top1.avg,top5.avg,prec.avg,recall.avg,f1.avg,losses.avg


if __name__ == '__main__':
    main()
