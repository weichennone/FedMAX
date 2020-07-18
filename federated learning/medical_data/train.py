import os
import argparse
from datetime import datetime
import copy

import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms

from lib.dataset import Dataset
from lib.models.model_factory import get_model
from lib.utils import *
from lib.metrics import *
from lib.losses import *
from lib.optimizers import *
from lib.preprocess import preprocess
from fed import fedavg
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                        help='model architecture: ' +
                        ' (default: resnet34)')
    parser.add_argument('--freeze_bn', default=True, type=str2bool)
    parser.add_argument('--dropout_p', default=0, type=float)
    parser.add_argument('--loss', default='MSELoss',
                        choices=['CrossEntropyLoss', 'FocalLoss', 'MSELoss', 'multitask'])
    parser.add_argument('--reg_coef', default=1.0, type=float)
    parser.add_argument('--cls_coef', default=0.1, type=float)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--img_size', default=288, type=int,
                        help='input image size (default: 288)')
    parser.add_argument('--input_size', default=256, type=int,
                        help='input image size (default: 256)')
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--pred_type', default='regression',
                        choices=['classification', 'regression', 'multitask'])
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.5, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # preprocessing
    parser.add_argument('--scale_radius', default=True, type=str2bool)
    parser.add_argument('--normalize', default=False, type=str2bool)
    parser.add_argument('--padding', default=False, type=str2bool)
    parser.add_argument('--remove', default=False, type=str2bool)

    # data augmentation
    parser.add_argument('--rotate', default=True, type=str2bool)
    parser.add_argument('--rotate_min', default=-180, type=int)
    parser.add_argument('--rotate_max', default=180, type=int)
    parser.add_argument('--rescale', default=True, type=str2bool)
    parser.add_argument('--rescale_min', default=0.8889, type=float)
    parser.add_argument('--rescale_max', default=1.0, type=float)
    parser.add_argument('--shear', default=True, type=str2bool)
    parser.add_argument('--shear_min', default=-36, type=int)
    parser.add_argument('--shear_max', default=36, type=int)
    parser.add_argument('--translate', default=False, type=str2bool)
    parser.add_argument('--translate_min', default=0, type=float)
    parser.add_argument('--translate_max', default=0, type=float)
    parser.add_argument('--flip', default=True, type=str2bool)
    parser.add_argument('--contrast', default=True, type=str2bool)
    parser.add_argument('--contrast_min', default=0.9, type=float)
    parser.add_argument('--contrast_max', default=1.1, type=float)
    parser.add_argument('--random_erase', default=False, type=str2bool)
    parser.add_argument('--random_erase_prob', default=0.5, type=float)
    parser.add_argument('--random_erase_sl', default=0.02, type=float)
    parser.add_argument('--random_erase_sh', default=0.4, type=float)
    parser.add_argument('--random_erase_r', default=0.3, type=float)

    # dataset
    parser.add_argument('--train_dataset',
                        default='diabetic_retinopathy + aptos2019')
    parser.add_argument('--cv', default=True, type=str2bool)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--remove_duplicate', default=False, type=str2bool)
    parser.add_argument('--class_aware', default=False, type=str2bool)

    # federated learning
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--iid', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # pseudo label
    parser.add_argument('--pretrained_model')
    parser.add_argument('--pseudo_labels')

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, lr):
    losses = AverageMeter()
    scores = AverageMeter()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif args.optimizer == 'RAdam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    model.train()

    def cal_uniform_act(out):
        shape = out.size()
        zero_mat = torch.zeros(shape).cuda()
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        kldiv = nn.KLDivLoss(reduce=True)
        cost = args.beta * kldiv(logsoftmax(out), softmax(zero_mat))
        return cost

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        activation = model.features(input)
        output = model.logits(activation)
        if args.pred_type == 'classification':
            loss = criterion(output, target)
        elif args.pred_type == 'regression':
            loss = criterion(output.view(-1), target.float())
        elif args.pred_type == 'multitask':
            loss = args.reg_coef * criterion['regression'](output[:, 0], target.float()) + \
                   args.cls_coef * criterion['classification'](output[:, 1:], target)
            output = output[:, 0].unsqueeze(1)

        loss += cal_uniform_act(activation.view(activation.size(0), -1))
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.pred_type == 'regression' or args.pred_type == 'multitask':
            thrs = [0.5, 1.5, 2.5, 3.5]
            output[output < thrs[0]] = 0
            output[(output >= thrs[0]) & (output < thrs[1])] = 1
            output[(output >= thrs[1]) & (output < thrs[2])] = 2
            output[(output >= thrs[2]) & (output < thrs[3])] = 3
            output[output >= thrs[3]] = 4
        if args.pseudo_labels is not None:
            thrs = [0.5, 1.5, 2.5, 3.5]
            target[target < thrs[0]] = 0
            target[(target >= thrs[0]) & (target < thrs[1])] = 1
            target[(target >= thrs[1]) & (target < thrs[2])] = 2
            target[(target >= thrs[2]) & (target < thrs[3])] = 3
            target[target >= thrs[3]] = 4
        score = quadratic_weighted_kappa(output, target)

        losses.update(loss.item(), input.size(0))
        scores.update(score, input.size(0))

    return losses.avg, scores.avg, model.state_dict()


def test(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()
    scores_f1 = AverageMeter()
    correct = 0
    # switch to evaluate mode
    model.eval()
    outputs = []
    targets = []
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            if args.pred_type == 'classification':
                loss = criterion(output, target)
            elif args.pred_type == 'regression':
                loss = criterion(output.view(-1), target.float())
            elif args.pred_type == 'multitask':
                loss = args.reg_coef * criterion['regression'](output[:, 0], target.float()) + \
                       args.cls_coef * criterion['classification'](output[:, 1:], target)
                output = output[:, 0].unsqueeze(1)

            if args.pred_type == 'regression' or args.pred_type == 'multitask':
                thrs = [0.5, 1.5, 2.5, 3.5]
                output[output < thrs[0]] = 0
                output[(output >= thrs[0]) & (output < thrs[1])] = 1
                output[(output >= thrs[1]) & (output < thrs[2])] = 2
                output[(output >= thrs[2]) & (output < thrs[3])] = 3
                output[output >= thrs[3]] = 4
            score = quadratic_weighted_kappa(output, target)
            score_f1 = f1_micro(output, target)

            outputs.append(output.detach().clone())
            targets.append(target.clone())
            output = output.type(torch.cuda.LongTensor)
            correct += output.eq(target.data.view_as(output)).long().cpu().sum()

            losses.update(loss.item(), input.size(0))
            scores.update(score, input.size(0))
            scores_f1.update(score_f1, input.size(0))
    outputs = torch.cat(tuple(outputs))
    targets = torch.cat(tuple(targets))
    confusion_m = confusion_mat(outputs, targets)
    accuracy = 100.00 * float(correct) / len(val_loader.dataset)

    return losses.avg, scores.avg, scores_f1.avg, accuracy, confusion_m


def split_dataset(all_labels, usr_num, iid):
    ratio = 0.15
    all_ind = np.argsort(all_labels)

    unique, counts = np.unique(all_labels, return_counts=True)
    label_dict = dict(zip(unique, counts))

    train_ind_dict = {}
    test_ind = []
    start_ind = 0
    train_total = 0
    for k in label_dict:
        test_ind.append(all_ind[start_ind: start_ind + int(label_dict[k] * ratio)])
        train_ind_dict[k] = all_ind[start_ind + int(label_dict[k] * ratio): start_ind + label_dict[k]]
        start_ind += label_dict[k]
        train_total += len(train_ind_dict[k])
    test_ind = np.concatenate(tuple(test_ind))

    user_ind_dict = {i:[] for i in range(usr_num)}
    if iid == 0: # IID case
        for k in label_dict:
            ind_all = np.random.permutation(len(train_ind_dict[k]))
            ind_all_list = np.array_split(ind_all, usr_num)
            for i in range(usr_num):
                user_ind_dict[i].append(train_ind_dict[k][ind_all_list[i]])
        for i in range(usr_num):
            user_ind_dict[i] = np.concatenate(tuple(user_ind_dict[i]))
    else:
        indiv_sample = int(train_total / usr_num)
        extra_sample = train_total % usr_num
        num_samples = {}
        for i in range(usr_num):
            num_samples[i] = indiv_sample + 1 if i < extra_sample else indiv_sample
        for k in label_dict:
            for i in range(usr_num):
                num = num_samples[i] if k == unique[-1] else np.random.randint(num_samples[i])
                if num < len(train_ind_dict[k]):
                    inds = np.random.choice(len(train_ind_dict[k]), num, replace=False)
                    user_ind_dict[i].append(train_ind_dict[k][inds])
                    train_ind_dict[k] = np.delete(train_ind_dict[k], inds)
                    num_samples[i] -= num
                elif len(train_ind_dict[k]) > 0:
                    num_samples[i] -= len(train_ind_dict[k])
                    user_ind_dict[i].append(train_ind_dict[k])
                    train_ind_dict[k] = np.delete(train_ind_dict[k], np.arange(len(train_ind_dict[k])))
            
        for i in range(usr_num):
            if num_samples[i] > 0:
                inds = np.random.choice(len(train_ind_dict[0]), num_samples[i], replace=False)
                user_ind_dict[i].append(train_ind_dict[0][inds])
                train_ind_dict[0] = np.delete(train_ind_dict[0], inds)
            user_ind_dict[i] = np.concatenate(tuple(user_ind_dict[i]))
            
    return user_ind_dict, test_ind[np.random.permutation(len(test_ind))]


def main():
    args = parse_args()
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    if args.name is None:
        args.name = '%s_%s' % (args.arch, datetime.now().strftime('%m%d%H'))

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('- %s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('- %s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'FocalLoss':
        criterion = FocalLoss().cuda()
    elif args.loss == 'MSELoss':
        criterion = nn.MSELoss().cuda()
    elif args.loss == 'multitask':
        criterion = {
            'classification': nn.CrossEntropyLoss().cuda(),
            'regression': nn.MSELoss().cuda(),
        }
    else:
        raise NotImplementedError

    if args.pred_type == 'classification':
        num_outputs = 5
    elif args.pred_type == 'regression':
        num_outputs = 1
    elif args.loss == 'multitask':
        num_outputs = 6
    else:
        raise NotImplementedError

    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomAffine(
            degrees=(args.rotate_min, args.rotate_max) if args.rotate else 0,
            translate=(args.translate_min, args.translate_max) if args.translate else None,
            scale=(args.rescale_min, args.rescale_max) if args.rescale else None,
            shear=(args.shear_min, args.shear_max) if args.shear else None,
        ),
        transforms.CenterCrop(args.input_size),
        transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
        transforms.RandomVerticalFlip(p=0.5 if args.flip else 0),
        transforms.ColorJitter(
            brightness=0,
            contrast=args.contrast,
            saturation=0,
            hue=0),
        RandomErase(
            prob=args.random_erase_prob if args.random_erase else 0,
            sl=args.random_erase_sl,
            sh=args.random_erase_sh,
            r=args.random_erase_r),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # data loading code
    if 'diabetic_retinopathy' in args.train_dataset:
        diabetic_retinopathy_dir = preprocess(
            'diabetic_retinopathy',
            args.img_size,
            scale=args.scale_radius,
            norm=args.normalize,
            pad=args.padding,
            remove=args.remove)
        diabetic_retinopathy_df = pd.read_csv('inputs/diabetic-retinopathy-resized/trainLabels.csv')
        diabetic_retinopathy_img_paths = \
            diabetic_retinopathy_dir + '/' + diabetic_retinopathy_df['image'].values + '.jpeg'
        diabetic_retinopathy_labels = diabetic_retinopathy_df['level'].values

    if 'aptos2019' in args.train_dataset:
        aptos2019_dir = preprocess(
            'aptos2019',
            args.img_size,
            scale=args.scale_radius,
            norm=args.normalize,
            pad=args.padding,
            remove=args.remove)
        aptos2019_df = pd.read_csv('inputs/train.csv')
        aptos2019_img_paths = aptos2019_dir + '/' + aptos2019_df['id_code'].values + '.png'
        aptos2019_labels = aptos2019_df['diagnosis'].values

    if 'chestxray' in args.train_dataset:
        chestxray_dir = preprocess(
            'chestxray',
            args.img_size,
            scale=args.scale_radius,
            norm=args.normalize,
            pad=args.padding,
            remove=args.remove)

        chestxray_img_paths = []
        chestxray_labels = []
        normal_cases = glob('chest_xray/chest_xray/train/NORMAL/*.jpeg')
        pneumonia_cases = glob('chest_xray/chest_xray/train/PNEUMONIA/*.jpeg')
        for nor in normal_cases:
            p = nor.split('/')[-1]
            chestxray_img_paths.append(chestxray_dir + '/' + p)
            chestxray_labels.append(0)
        for abn in pneumonia_cases:
            p = abn.split('/')[-1]
            chestxray_img_paths.append(chestxray_dir + '/' + p)
            chestxray_labels.append(1)

        normal_cases = glob('chest_xray/chest_xray/test/NORMAL/*.jpeg')
        pneumonia_cases = glob('chest_xray/chest_xray/test/PNEUMONIA/*.jpeg')
        for nor in normal_cases:
            p = nor.split('/')[-1]
            chestxray_img_paths.append(chestxray_dir + '/' + p)
            chestxray_labels.append(0)
        for abn in pneumonia_cases:
            p = abn.split('/')[-1]
            chestxray_img_paths.append(chestxray_dir + '/' + p)
            chestxray_labels.append(1)

        normal_cases = glob('chest_xray/chest_xray/val/NORMAL/*.jpeg')
        pneumonia_cases = glob('chest_xray/chest_xray/val/PNEUMONIA/*.jpeg')
        for nor in normal_cases:
            p = nor.split('/')[-1]
            chestxray_img_paths.append(chestxray_dir + '/' + p)
            chestxray_labels.append(0)
        for abn in pneumonia_cases:
            p = abn.split('/')[-1]
            chestxray_img_paths.append(chestxray_dir + '/' + p)
            chestxray_labels.append(1)

        chestxray_img_paths = np.array(chestxray_img_paths)
        chestxray_labels = np.array(chestxray_labels)

    if args.train_dataset == 'aptos2019':
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=41)
        img_paths = []
        labels = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(aptos2019_img_paths, aptos2019_labels)):
            img_paths.append((aptos2019_img_paths[train_idx], aptos2019_img_paths[val_idx]))
            labels.append((aptos2019_labels[train_idx], aptos2019_labels[val_idx]))
    elif args.train_dataset == 'diabetic_retinopathy':
        img_paths = [(diabetic_retinopathy_img_paths, aptos2019_img_paths)]
        labels = [(diabetic_retinopathy_labels, aptos2019_labels)]
    elif 'diabetic_retinopathy' in args.train_dataset and 'aptos2019' in args.train_dataset:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=41)
        img_paths = []
        labels = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(aptos2019_img_paths, aptos2019_labels)):
            img_paths.append((np.hstack((aptos2019_img_paths[train_idx], diabetic_retinopathy_img_paths)), aptos2019_img_paths[val_idx]))
            labels.append((np.hstack((aptos2019_labels[train_idx], diabetic_retinopathy_labels)), aptos2019_labels[val_idx]))
    else:
        raise NotImplementedError

    # FL setting: separate data into users
    if 'diabetic_retinopathy' in args.train_dataset and 'aptos2019' in args.train_dataset:
        combined_paths = np.hstack((aptos2019_img_paths, diabetic_retinopathy_img_paths))
        combined_labels = np.hstack((aptos2019_labels, diabetic_retinopathy_labels))
    elif 'chestxray' in args.train_dataset:
        combined_paths = chestxray_img_paths
        combined_labels = chestxray_labels
    else:
        raise NotImplementedError
    user_ind_dict, ind_test = split_dataset(combined_labels, args.num_users, args.iid)

    model = get_model(model_name=args.arch,
                      num_outputs=num_outputs,
                      freeze_bn=args.freeze_bn,
                      dropout_p=args.dropout_p)
    model = model.cuda()
    test_set = Dataset(
        combined_paths[ind_test],
        combined_labels[ind_test],
        transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)

    test_acc = []
    test_scores = []
    test_scores_f1 = []
    lr = args.lr
    for epoch in range(args.epochs):

        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))
        weight_list = []
        selected_ind = np.random.choice(args.num_users, int(args.num_users / 10), replace=False)
        for i in selected_ind:
            print('user: %d' % (i + 1))
            train_set = Dataset(
                combined_paths[user_ind_dict[i]],
                combined_labels[user_ind_dict[i]],
                transform=train_transform)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=False if args.class_aware else True,
                num_workers=4,
                sampler=sampler if args.class_aware else None)

            # train for one epoch
            train_loss, train_score, ret_w = train(
                args, train_loader, copy.deepcopy(model), criterion, lr)
            weight_list.append(ret_w)
            print('loss %.4f - score %.4f'
                  % (train_loss, train_score))
        weights = fedavg(weight_list)
        model.load_state_dict(weights)
        test_loss, test_score, test_scoref1, accuracy, confusion_matrix = test(
            args, test_loader, copy.deepcopy(model), criterion)
        print('loss %.4f - score %.4f - accuracy %.4f'
              % (test_loss, test_score, accuracy))
        test_acc.append(accuracy)
        test_scores.append(test_score)
        test_scores_f1.append(test_scoref1)
        lr *= 0.992

    np.savez('./accuracy-xray-iid'+str(args.iid)+'-' + str(args.epochs) +'-beta'+str(args.beta)+'-seed'+str(args.seed),
             acc=np.array(test_acc),
             score=np.array(test_scores),
             scoref1=np.array(test_scores_f1),
             confusion=confusion_matrix)


if __name__ == '__main__':

    main()
