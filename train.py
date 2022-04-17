# -*- coding: utf-8 -*-
from numpy import reshape
import config
from utils import meter
from torch import nn
from torch import optim
from models import Net
from torch.utils.data import DataLoader
from center_loss import CrossModalCenterLoss
from datasets import *
import argparse
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./log')  # 实例化writer
name_list = [
    'night_stand', 'range_hood', 'plant', 'chair', 'tent', 'curtain', 'piano',
    'dresser', 'desk', 'bed', 'sink', 'laptop', 'flower_pot', 'car', 'stool',
    'vase', 'monitor', 'airplane', 'stairs', 'glass_box', 'bottle', 'guitar',
    'cone', 'toilet', 'bathtub', 'wardrobe', 'radio', 'person', 'xbox', 'bowl',
    'cup', 'door', 'tv_stand', 'mantel', 'sofa', 'keyboard', 'bookshelf',
    'bench', 'table', 'lamp'
]


def get_w2v_class(label_name_list):
    model = KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin.gz', binary=True)
    w2v_class = []
    num = 0
    for i, voca in enumerate(label_name_list):
        try:
            w2v_class.append(np.array(model[voca]))
            num += 1
            # print(voca)
        except:
            try:
                list_vec = []
                list_word = voca.split("_")
                for j, word in enumerate(list_word):
                    word_vec = np.array(model[word])
                    list_vec.append(word_vec)
                class_vec = np.mean(list_vec, axis=0)
                w2v_class.append(class_vec)
                num += 1
                # print(voca)
            except:
                print(voca)
    return w2v_class


w2v_classes_list = get_w2v_class(name_list)


def train(train_loader, net, criterion, cmc_criterion, mse_criterion,
          optimizer, optimizer_centloss, epoch, cmc_loss_args):
    """
    train for one epoch on the training set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    losses = meter.AverageValueMeter()
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    # training mode
    net.train()

    for i, (views, pcs, labels, w2v_classes) in enumerate(train_loader):
        batch_time.reset()
        views = views.to(device=config.device)
        pcs = pcs.to(device=config.device)
        labels = labels.to(device=config.device)
        w2v_classes = w2v_classes.to(device=config.device)
        w2v_classes = w2v_classes.squeeze()
        # preds = net(pcs, views)  # bz x C x H x W
        preds, fts, pc, mv = net(pcs, views, get_fea=True)  # bz x C x H x W

        # ce loss
        ce_loss = criterion(preds, labels)
        # cross modal center loss
        cmc_loss = cmc_criterion(torch.cat((mv, pc), dim=0),
                                 torch.cat((labels, labels), dim=0))

        # DSCMR loss
        dscmr_loss = calc_loss(mv, pc, fts, fts, w2v_classes, w2v_classes,
                               1e-3, 1e-1)

        prec.add(preds.detach(), labels.detach())
        # mse loss
        mse_loss = mse_criterion(mv, pc)

        losses.add(ce_loss.item())  # batchsize
        losses.add(cmc_loss.item())
        losses.add(mse_loss.item())

        # losses.add(dscmr_loss.item())

        # weighted the three losses as final loss
        loss = ce_loss + cmc_loss_args.weight_center * cmc_loss + 0.1 * ``mse_loss`` + 0.1 * dscmr_loss
        # loss = ce_loss + cmc_loss_args.weight_center * cmc_loss + 0.1 * mse_loss
        optimizer.zero_grad()
        optimizer_centloss.zero_grad()

        # loss = criterion(preds, labels)

        # prec.add(preds.detach(), labels.detach())
        # losses.add(loss.item())  # batchsize

        # optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # by doing so, weight_cent would not impact on the learning of centers
        for param in cmc_criterion.parameters():
            param.grad.data *= (1. / cmc_loss_args.weight_center)
        optimizer_centloss.step()

        # tensorboard
        writer.add_scalar(tag='Loss/ce_loss',
                          scalar_value=ce_loss.item(),
                          global_step=epoch)
        writer.add_scalar(tag='Loss/cmc_loss',
                          scalar_value=cmc_loss.item(),
                          global_step=epoch)
        writer.add_scalar(tag='Loss/mse_loss',
                          scalar_value=mse_loss.item(),
                          global_step=epoch)
        writer.add_scalar(tag='Loss/dscmr_loss',
                          scalar_value=dscmr_loss.item(),
                          global_step=epoch)

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Loss {losses.value()[0]:.4f} \t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    print(f'prec at epoch {epoch}: {prec.value(1)} ')


# def validate(val_loader, net, epoch):
#     """
#     validation for one epoch on the val set
#     """
#     batch_time = meter.TimeMeter(True)
#     data_time = meter.TimeMeter(True)
#     prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
#     retrieval_map = meter.RetrievalMAPMeter()

#     # testing mode
#     net.eval()

#     total_seen_class = [0 for _ in range(40)]
#     total_right_class = [0 for _ in range(40)]
#     num = 0
#     for i, (views, pcs, labels, w2v_classes) in enumerate(val_loader):
#         batch_time.reset()

#         views = views.to(device=config.device)
#         pcs = pcs.to(device=config.device)
#         labels = labels.to(device=config.device)
#         # w2v_classes_list = w2v_classes_list.to(device=config.device)
#         w2v_classes = w2v_classes.squeeze()
#         # w2v_classes_list = w2v_classes_list.squeeze()
#         # w2v_classes_list = np.array(w2v_classes_list)
#         # w2v_classes_list = reshape(w2v_classes_list, (40,-1))
#         # for k in range(40):
#         #     w2v_classes_list[i] = torch.from_numpy(w2v_classes_list[k])
#         # w2v_classes_list = torch.from_numpy(w2v_classes_list.astype(float))
#         preds, fts, pc, mv = net(pcs, views, get_fea=True)  # bz x C x H x W
#         # prec.add(preds.data, labels.data)
#         w2v_classes_labels = []
#         for item in w2v_classes_list:
#             w2v = item[0]
#             w2v = torch.tensor(w2v)
#             w2v = w2v.squeeze()
#             w2v_classes_labels.append(w2v)
#         # dist = F.cosine_similarity(fts, w2v_classes, dim=1)
#         # torch.tensor(w2v_classes_labels)
#         # prec.add(preds.data, labels.data)
#         # prec.add(preds.data, w2v_classes.data)
#         retrieval_map.add(fts.detach() / torch.norm(fts.detach(), 2, 1, True), labels.detach())

#         for j in range(views.size(0)):
#             pred = preds.data[j]
#             pred = pred.view(1, len(pred))
#             # pred = pred.repeat(40 ,1)
#             dist = []
#             for item in w2v_classes_list:
#                 w2v = item[0]
#                 w2v = torch.tensor(w2v)
#                 # w2v.to(device=config.device)
#                 # w2v = w2v.view(1, len(w2v))
#                 dis = F.cosine_similarity(pred.cpu(), w2v, dim=1)
#                 dist.append(dis[0])
#             # dist = F.cosine_similarity(pred.cpu().data, w2v_classes_list, dim=1)
#             # val, idx = torch.max(dist, 0)
#             idx = np.argmax(dist)
#             if idx == labels.cpu()[j]:
#                 num = num + 1
#             # total_seen_class[labels.data[j]] += 1
#             # total_right_class[labels.data[j]] += (idx == labels.cpu()[j])
#             # total_right_class[labels.data[j]] += (np.argmax(preds.cpu().data, 1)[j] == labels.cpu()[j])

#             # total_right_class[labels.data[j]] += (np.argmax(preds.cpu().data, 1)[j] == labels.cpu()[j])
#     acc = round(num / 246, 5)
#     #     if i % config.print_freq == 0:
#     #         print(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
#     #               f'Batch Time {batch_time.value():.3f}\t'
#     #               f'Epoch Time {data_time.value():.3f}\t'
#     #               f'Prec@1 {prec.value(1):.3f}\t')


#     mAP = retrieval_map.mAP()
#     # print(f' instance accuracy at epoch {epoch}: {prec.value(1)} ')
#     # print(f' mean class accuracy at epoch {epoch}: {(np.mean(np.array(total_right_class)/np.array(total_seen_class,dtype=np.float)))} ')
#     # print(f' map at epoch {epoch}: {mAP} ')
#     # return prec.value(1), mAP
#     print(f' instance accuracy at epoch {epoch}: {acc} ')
#     print(f' val_loader.dataset at epoch {epoch}: {len(val_loader.dataset)} ')
#     return acc, mAP
def validate(val_loader, net, epoch):
    """
    validation for one epoch on the val set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    retrieval_map = meter.RetrievalMAPMeter()

    # testing mode
    net.eval()

    total_seen_class = [0 for _ in range(40)]
    total_right_class = [0 for _ in range(40)]

    for i, (views, pcs, labels, w2v_classes) in enumerate(val_loader):
        batch_time.reset()

        views = views.to(device=config.device)
        pcs = pcs.to(device=config.device)
        labels = labels.to(device=config.device)

        preds, fts, pc, mv = net(pcs, views, get_fea=True)  # bz x C x H x W
        # prec.add(preds.data, labels.data)

        prec.add(preds.data, labels.data)
        retrieval_map.add(fts.detach() / torch.norm(fts.detach(), 2, 1, True),
                          labels.detach())
        for j in range(views.size(0)):
            total_seen_class[labels.data[j]] += 1
            total_right_class[labels.data[j]] += (np.argmax(
                preds.cpu().data, 1)[j] == labels.cpu()[j])

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    mAP = retrieval_map.mAP()
    print(f' instance accuracy at epoch {epoch}: {prec.value(1)} ')
    print(
        f' mean class accuracy at epoch {epoch}: {(np.mean(np.array(total_right_class)/np.array(total_seen_class,dtype=np.float)))} '
    )
    print(f' map at epoch {epoch}: {mAP} ')
    return prec.value(1), mAP


def save_ckpt(epoch,
              epoch_pc,
              epoch_all,
              best_prec1,
              net,
              optimizer_pc,
              optimizer_all,
              training_conf=config.pv_net):
    ckpt = dict(epoch=epoch,
                epoch_pc=epoch_pc,
                epoch_all=epoch_all,
                best_prec1=best_prec1,
                model=net.module.state_dict(),
                optimizer_pc=optimizer_pc.state_dict(),
                optimizer_all=optimizer_all.state_dict(),
                training_conf=training_conf)
    torch.save(ckpt, config.pv_net.ckpt_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Main", )
    parser.add_argument("-batch_size",
                        '-b',
                        type=int,
                        default=32,
                        help="Batch size")
    parser.add_argument('-gpu', '-g', type=str, default=None, help='GPUS used')
    parser.add_argument("-epochs",
                        '-e',
                        type=int,
                        default=None,
                        help="Number of epochs to train for")
    return parser.parse_args()


# DSCMR


def calc_label_sim(label_1, label_2):
    # label_1 = label_1.view(len(label_1), 1)
    # label_2 = label_2.view(len(label_2), 1)
    Sim = label_1.squeeze().float().mm(label_2.squeeze().float().t())
    return Sim


def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict,
              labels_1, labels_2, alpha, beta):
    term1 = ((view1_predict - labels_1.float())**2).sum(1).sqrt().mean() + (
        (view2_predict - labels_2.float())**2).sum(1).sqrt().mean()

    cos = lambda x, y: x.mm(y.t()) / ((x**2).sum(1, keepdim=True).sqrt().mm(
        (y**2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()
    term21 = ((1 + torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term22 = ((1 + torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term23 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23

    term3 = ((view1_feature - view2_feature)**2).sum(1).sqrt().mean()

    im_loss = term1 + alpha * term2 + beta * term3
    # im_loss = alpha * term2 + beta * term3
    return im_loss


def main(cmc_loss_args):
    print('Training Process\nInitializing...\n')
    config.init_env()
    args = parse_args()

    total_batch_sz = config.pv_net.train.batch_sz * len(
        config.available_gpus.split(','))
    total_epoch = config.pv_net.train.max_epoch

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        total_batch_sz = config.pv_net.train.batch_sz * len(
            args.gpu.split(','))
    if args.epochs is not None:
        total_epoch = args.epochs

    train_dataset = pc_view_data(config.pv_net.pc_root,
                                 config.pv_net.view_root,
                                 status=STATUS_TRAIN,
                                 base_model_name=config.base_model_name)
    val_dataset = pc_view_data(config.pv_net.pc_root,
                               config.pv_net.view_root,
                               status=STATUS_TEST,
                               base_model_name=config.base_model_name)

    train_loader = DataLoader(train_dataset,
                              batch_size=total_batch_sz,
                              num_workers=config.num_workers,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=total_batch_sz,
                            num_workers=config.num_workers,
                            shuffle=True)

    best_prec1 = 0
    best_map = 0
    resume_epoch = 0

    epoch_pc_view = 0
    epoch_pc = 0

    # create model
    net = Net()
    net = net.to(device=config.device)
    net = nn.DataParallel(net)

    # optimizer
    fc_param = [{
        'params': v
    } for k, v in net.named_parameters() if 'fusion' in k]
    if config.pv_net.train.optim == 'Adam':
        optimizer_fc = optim.Adam(
            fc_param,
            config.pv_net.train.fc_lr,
            weight_decay=config.pv_net.train.weight_decay)

        optimizer_all = optim.Adam(
            net.parameters(),
            config.pv_net.train.all_lr,
            weight_decay=config.pv_net.train.weight_decay)
    elif config.pv_net.train.optim == 'SGD':
        optimizer_fc = optim.SGD(fc_param,
                                 config.pv_net.train.fc_lr,
                                 momentum=config.pv_net.train.momentum,
                                 weight_decay=config.pv_net.train.weight_decay)

        optimizer_all = optim.SGD(
            net.parameters(),
            config.pv_net.train.all_lr,
            momentum=config.pv_net.train.momentum,
            weight_decay=config.pv_net.train.weight_decay)
    else:
        raise NotImplementedError
    print(f'use {config.pv_net.train.optim} optimizer')
    print(f'Sclae:{net.module.n_scale} ')

    if config.pv_net.train.resume:
        print(f'loading pretrained model from {config.pv_net.ckpt_file}')
        checkpoint = torch.load(config.pv_net.ckpt_file)
        state_dict = checkpoint['model']
        net.module.load_state_dict(checkpoint['model'])
        optimizer_fc.load_state_dict(checkpoint['optimizer_pc'])
        optimizer_all.load_state_dict(checkpoint['optimizer_all'])
        best_prec1 = checkpoint['best_prec1']
        epoch_pc_view = checkpoint['epoch_all']
        epoch_pc = checkpoint['epoch_pc']
        if config.pv_net.train.resume_epoch is not None:
            resume_epoch = config.pv_net.train.resume_epoch
        else:
            resume_epoch = max(checkpoint['epoch_pc'], checkpoint['epoch_all'])

    if config.pv_net.train.iter_train == False:
        print('No iter')
        lr_scheduler_fc = torch.optim.lr_scheduler.StepLR(optimizer_fc, 5, 0.3)
        lr_scheduler_all = torch.optim.lr_scheduler.StepLR(
            optimizer_all, 5, 0.3)
    else:
        print('iter')
        lr_scheduler_fc = torch.optim.lr_scheduler.StepLR(optimizer_fc, 6, 0.3)
        lr_scheduler_all = torch.optim.lr_scheduler.StepLR(
            optimizer_all, 6, 0.3)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=config.device)

    # cross modal center loss
    cmc_criterion = CrossModalCenterLoss(num_classes=config.pv_net.num_classes,
                                         feat_dim=1024,
                                         use_gpu=True)
    optimizer_centloss = optim.SGD(cmc_criterion.parameters(),
                                   lr=cmc_loss_args.lr_center)
    cmc_criterion = cmc_criterion.to(device=config.device)
    # mse loss
    mse_criterion = nn.MSELoss()
    mse_criterion = mse_criterion.to(device=config.device)
    iteration = 0
    for epoch in range(resume_epoch, total_epoch):

        if config.pv_net.train.iter_train == True:
            if epoch < 12:
                lr_scheduler_fc.step(epoch=epoch_pc)
                print(lr_scheduler_fc.get_last_lr())

                if (epoch_pc + 1) % 3 == 0:
                    print('train score block')
                    for m in net.module.parameters():
                        m.reqires_grad = False
                    net.module.fusion_conv1.requires_grad = True
                else:
                    print('train all fc block')
                    for m in net.module.parameters():
                        m.reqires_grad = True

                # train(train_loader, net, criterion, optimizer_fc, epoch)
                train(train_loader, net, criterion, cmc_criterion,
                      mse_criterion, optimizer_fc, optimizer_centloss, epoch,
                      cmc_loss_args)
                # update the learning rate of the center loss
                if (iteration % cmc_loss_args.lr_step) == 0:
                    lr_center = cmc_loss_args.lr_center * (0.1**(
                        iteration // cmc_loss_args.lr_step))
                    print('New  Center LR:     ' + str(lr_center))
                    for param_group in optimizer_centloss.param_groups:
                        param_group['lr'] = lr_center

                iteration + 1

                epoch_pc += 1

            else:
                lr_scheduler_all.step(epoch=epoch_pc_view)
                print(lr_scheduler_all.get_last_lr())

                if (epoch_pc_view + 1) % 3 == 0:
                    print('train score block')
                    for m in net.module.parameters():
                        m.reqires_grad = False
                    net.module.fusion_conv1.requires_grad = True
                else:
                    print('train all block')
                    for m in net.module.parameters():
                        m.reqires_grad = True

                # train(train_loader, net, criterion, optimizer_all, epoch)
                train(train_loader, net, criterion, cmc_criterion,
                      mse_criterion, optimizer_all, optimizer_centloss, epoch,
                      cmc_loss_args)
                # update the learning rate of the center loss
                if (iteration % cmc_loss_args.lr_step) == 0:
                    lr_center = cmc_loss_args.lr_center * (0.1**(
                        iteration // cmc_loss_args.lr_step))
                    print('New  Center LR:     ' + str(lr_center))
                    for param_group in optimizer_centloss.param_groups:
                        param_group['lr'] = lr_center

                iteration + 1

                epoch_pc_view += 1

        else:
            if epoch < 10:
                lr_scheduler_fc.step(epoch=epoch_pc)
                print(lr_scheduler_fc.get_last_lr())
                # train(train_loader, net, criterion, optimizer_fc, epoch)
                train(train_loader, net, criterion, cmc_criterion,
                      mse_criterion, optimizer_fc, optimizer_centloss, epoch,
                      cmc_loss_args)
                # update the learning rate of the center loss
                if (iteration % cmc_loss_args.lr_step) == 0:
                    lr_center = cmc_loss_args.lr_center * (0.1**(
                        iteration // cmc_loss_args.lr_step))
                    print('New  Center LR:     ' + str(lr_center))
                    for param_group in optimizer_centloss.param_groups:
                        param_group['lr'] = lr_center

                iteration + 1

                epoch_pc += 1

            else:
                lr_scheduler_all.step(epoch=epoch_pc_view)
                print(lr_scheduler_all.get_last_lr())
                # train(train_loader, net, criterion, optimizer_all, epoch)
                train(train_loader, net, criterion, cmc_criterion,
                      mse_criterion, optimizer_all, optimizer_centloss, epoch,
                      cmc_loss_args)
                # update the learning rate of the center loss
                if (iteration % cmc_loss_args.lr_step) == 0:
                    lr_center = cmc_loss_args.lr_center * (0.1**(
                        iteration // cmc_loss_args.lr_step))
                    print('New  Center LR:     ' + str(lr_center))
                    for param_group in optimizer_centloss.param_groups:
                        param_group['lr'] = lr_center

                iteration + 1

                epoch_pc_view += 1

        with torch.no_grad():
            prec1, retrieval_map = validate(val_loader, net, epoch)

        # save checkpoints
        if best_prec1 < prec1:
            best_prec1 = prec1
            save_ckpt(epoch, epoch_pc, epoch_pc_view, best_prec1, net,
                      optimizer_fc, optimizer_all)
        if best_map < retrieval_map:
            best_map = retrieval_map

        print('curr accuracy: ', prec1)
        print('best accuracy: ', best_prec1)
        print('best map: ', best_map)

    print('Train Finished!')


if __name__ == '__main__':
    # Training settings
    cmc_loss_parser = argparse.ArgumentParser(
        description=
        'Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')

    cmc_loss_parser.add_argument(
        '--lr_step',
        type=int,
        default=20000,
        help='how many iterations to decrease the learning rate')

    cmc_loss_parser.add_argument(
        '--lr_center',
        type=float,
        default=0.5,
        metavar='LR',
        help='learning rate for center loss (default: 0.5)')

    # loss
    cmc_loss_parser.add_argument('--weight_center',
                                 type=float,
                                 default=0.05,
                                 metavar='weight_center',
                                 help='weight center (default: 1.0)')

    cmc_loss_args = cmc_loss_parser.parse_args()
    main(cmc_loss_args)
