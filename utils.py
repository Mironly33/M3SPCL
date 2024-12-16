import numpy as np
import random
import logging
import torch
import shutil
import os
import sklearn.metrics as metrics

def normalize(x):
    x = (x-np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0)-np.min(x, axis=0)), (x.shape[0], 1))
    return x


def random_index(n_all, n_train, seed):
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)   
    train_idx = random_idx[0:n_train]                 
    test_idx = random_idx[n_train:n_all]              
    return train_idx, test_idx


def TT_split(n_all, seed):
    '''
    split data into training, testing dataset
    '''
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)  
    train_num = np.ceil(0.7 * n_all).astype(np.int)   
    train_idx = random_idx[0:train_num]                      
    test_num = np.floor(0.3 * n_all).astype(np.int)       
    test_idx = random_idx[-test_num:]   
    return train_idx, test_idx


def initLogging(logFilename):
   
    LOG_FORMAT = "%(asctime)s\tFile \"%(filename)s\",LINE %(lineno)-4d : %(levelname)-8s %(message)s"
   
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=logFilename, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    formatter = logging.Formatter(LOG_FORMAT);
    console = logging.StreamHandler();
    console.setLevel(logging.INFO);
    console.setFormatter(formatter);
    logging.getLogger('').addHandler(console);


def adjust_learning_rate(optimizer, epoch,args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learn_rate * (0.05 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print("\t".join(entries))
        log_message = "\t".join(entries)
        logging.info(log_message)  # 将信息写入日志文件

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    

def save_checkpoint(state, is_best,folder_name, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        file_model_best = os.path.join(folder_name, 'model_best.pth.tar')
        shutil.copyfile(filename,file_model_best)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def classification_metric(output,y_true, average='macro', decimals=6):
    # obtain the Y-pre and convert them to np.array
    with torch.no_grad():
        y_pred = output.argmax(dim=1, keepdim=True)
        y_pred = y_pred.view(-1).cpu().numpy()
        y_true = y_true.view(-1).cpu().numpy()
        # confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        # ACC
        accuracy = metrics.accuracy_score(y_true, y_pred)
        accuracy = np.round(accuracy, decimals)*100

        # precision
        precision = metrics.precision_score(y_true, y_pred, average=average) # 
        precision = np.round(precision, decimals)*100

        # recall
        recall = metrics.recall_score(y_true, y_pred, average=average)
        recall = np.round(recall, decimals)*100

        # F-score
        f_score = metrics.f1_score(y_true, y_pred, average=average)
        f_score = np.round(f_score, decimals)*100
        met = dict({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_score': f_score})
        # print('accuracy: {}, precision: {}, recall: {}, f_measure: {}'.format(accuracy, precision, recall, f_score))
    return met


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [d, m]    view0的feature:h0  m维  1024*10
        y: pytorch Variable, with shape [d, n]    view1的feature:h1  n维  1024*10
    Returns:
        dist: pytorch Variable, with shape [d, d]    
    """

    m, n = x.size(0), y.size(0)


    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)   
 
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()   
    dist = xx + yy
    
   
    dist.addmm_(1, -2, x, y.t())


    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def get_real_labels(label0,label1):  #N=2243 M=30   
    
    n0 = label0.shape[0]
    n1 = label1.shape[0]
    mask=torch.zeros((n0,n1))
   
    mask[label0[:, np.newaxis] == label1] = 1
    return  mask
  
def compute_pair_dist(fea,args):
    

    h = fea[:args.batch_size]
    h_PI = fea[args.batch_size:]
    pair_dist = euclidean_dist(h, h_PI)

    return pair_dist

def adjust_learning_rate(optimizer, epoch,args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learn_rate * (0.05 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr