import torch as t
from torch import nn
from data.triplet_input import TripleDataset
from data import TripleDataLoader
# import torch.utils.data as dataloader
from torch.utils.data import DataLoader 
from models.vgg import vgg16
from models.sketch_resnet import resnet50
# from utils.visualize import Visualizer
from torchnet.meter import AverageValueMeter
# import tqdm
from utils.extractor import Extractor
from sklearn.neighbors import NearestNeighbors
from torch.nn import DataParallel
from .TripletLoss import TripletLoss
from utils.test import Tester
import os
import numpy as np

# class Config(object):
#     def __init__(self):
#         return

# class TripletNet(object):

#     def __init__(self, opt):
#         # train config
#         self.photo_root = opt.photo_root
#         self.sketch_root = opt.sketch_root
#         self.batch_size = opt.batch_size
#         # self.device = opt.device
#         self.epochs = opt.epochs
#         self.lr = opt.lr
#         self.log_interval = opt.log_interval

#         # testing config
#         self.photo_test = opt.photo_test
#         self.sketch_test = opt.sketch_test
#         self.test = opt.test
#         self.test_f = opt.test_f

#         self.save_model = opt.save_model
#         self.save_dir = opt.save_dir

#         # vis
#         # self.vis = opt.vis
#         # self.env = opt.env

#         # fine_tune
#         # self.fine_tune = opt.fine_tune
#         # self.model_root = opt.model_root


#         # dataloader config
#         data_opt = Config()
#         data_opt.photo_root = opt.photo_root
#         data_opt.sketch_root = opt.sketch_root
#         data_opt.batch_size = opt.batch_size

#         self.dataloader_opt = data_opt

#         # triplet config
#         self.margin = opt.margin
#         self.p = opt.p

#         # feature extractor net
#         self.net = opt.net
#         self.cat = opt.cat

#     def _get_vgg16(self, pretrained=True):
#         model = MyVGG16(pretrained=pretrained)
#         return model

#     def _get_resnet50(self, pretrained=True):
#         model = resnet50(pretrained=pretrained)
#         model.fc = nn.Linear(in_features=2048, out_features=125)

#         return model

#     def train(self):
#         if self.net == 'vgg16':
#             photo_net = self._get_vgg16()
#             sketch_net = self._get_vgg16()
#         elif self.net == 'resnet50':
#             photo_net = self._get_resnet50()
#             sketch_net = self._get_resnet50()
#         # photo_net.to(device)
#         # sketch_net.to(device)

#         # if self.fine_tune:
#         #     photo_net_root = self.model_root
#         #     sketch_net_root = self.model_root.replace('photo', 'sketch')

#         #     photo_net.load_state_dict(t.load(photo_net_root, map_location=t.device('cpu')))
#         #     sketch_net.load_state_dict(t.load(sketch_net_root, map_location=t.device('cpu')))

#         print('net')
#         print(photo_net)

#         triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=self.p)

#         # optimizer
#         photo_optimizer = t.optim.Adam(photo_net.parameters(), lr=self.lr)
#         sketch_optimizer = t.optim.Adam(sketch_net.parameters(), lr=self.lr)

#         # # a learning rate scheduler which decreases the learning rate by
#         # # 0.5 every 2 epochs
#         # lr_scheduler = t.optim.lr_scheduler.StepLR(photo_optimizer,
#         #                                             step_size=2,
#         #                                             gamma=0.5)

#         # if self.vis:
#         #     vis = Visualizer(self.env)

#         # triplet_loss_meter = AverageValueMeter()
#         # sketch_cat_loss_meter = AverageValueMeter()
#         # photo_cat_loss_meter = AverageValueMeter()

#         data_loader = TripleDataLoader(self.dataloader_opt)
#         dataset = data_loader.load_data()
#         print(dataset.__len__())

#         for epoch in range(self.epochs):

#             print('---------------{0}---------------'.format(epoch))

#             photo_net.train()
#             sketch_net.train()
#             avg_loss = 0

#             for ii, data in enumerate(dataset):
#                 photo_optimizer.zero_grad()
#                 sketch_optimizer.zero_grad()

#                 anchor = data['A']
#                 anchor = anchor.unsqueeze(0)
#                 pos = data['P']
#                 pos = pos.unsqueeze(0)
#                 neg = data['N']
#                 neg = neg.unsqueeze(0)
#                 label = data['L']
#                 label1 = data['L1']

#                 # label = t.unsqueeze(t.tensor(label), 0)
                
#                 a_feature = sketch_net(anchor)
#                 p_feature= photo_net(pos)
#                 n_feature= photo_net(neg)

#                 loss = triplet_loss(a_feature, p_feature, n_feature)
#                 # loss = loss / self.batch_size

#                 loss.backward()

#                 # update param for model
#                 photo_optimizer.step()
#                 sketch_optimizer.step()

#                 print('[Train] Epoch: [{0}][{1}/{2}]\t'
#                         'Triplet loss  ({triplet_loss_meterr:.3f})\t'
#                         # 'Loss  ({losss:.4f})\t'
#                         # 'Sketch Loss ({acc:.4f})\t'
#                         .format(epoch + 1, ii + 1, len(dataset), triplet_loss_meterr=loss.item()))
#             if self.save_model:
#                 t.save(photo_net.state_dict(), self.save_dir + '/photo' + '/photo_' + self.net + '_%s.pth' % epoch)
#                 t.save(sketch_net.state_dict(), self.save_dir + '/sketch' + '/sketch_' + self.net + '_%s.pth' % epoch)



# import torch as t
# from torch import nn
# # from torchnet.meter import AverageValueMeter
# # import tqdm
# from torch.nn import DataParallel

# import os
# import numpy as np
from torch.autograd import Variable


class Config(object):
    def __init__(self):
        return


class TripletNet(object):

    def __init__(self, opt): 
        # train config
        self.photo_root = opt.photo_root
        self.sketch_root = opt.sketch_root
        self.batch_size = opt.batch_size
        # self.device = opt.device
        self.epochs = opt.epochs
        self.lr = opt.lr
        # testing config
        # self.photo_test = opt.photo_test
        # self.sketch_test = opt.sketch_test
        # self.test = opt.test
        # self.test_f = opt.test_f

#         self.save_model = opt.save_model
        self.save_dir = opt.save_dir

        # vis
#         self.vis = opt.vis
#         self.env = opt.env

        # fine_tune
#         self.fine_tune = opt.fine_tune
#         self.model_root = opt.model_root

        # dataloader config
        data_opt = Config()
        data_opt.photo_root = opt.photo_root
        data_opt.sketch_root = opt.sketch_root
        data_opt.batch_size = opt.batch_size

        self.dataloader_opt = data_opt

        # triplet config
        self.margin = opt.margin
        self.p = opt.p

        # feature extractor net
        self.net = opt.net
        self.cat = opt.cat

    def _get_vgg16(self, pretrained=True):
<<<<<<< HEAD
        model = MyVGG16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
=======
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
        return model
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658

        return model
    
    def _get_resnet50(self, pretrained=True):
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=10)

        return model

    def train(self):
<<<<<<< HEAD
        # photo_net = self._get_vgg16().cuda()
#         sketch_net = self._get_vgg16().cuda()
        
#         photo_net = self._get_vgg16()
#         sketch_net = self._get_vgg16()

        if self.net == 'resnet50':
            photo_net = self._get_resnet50().cuda()
        #     sketch_net = self._get_resnet50().cuda()
=======
        if self.net == 'vgg16':
            photo_net = self._get_vgg16()
            sketch_net = self._get_vgg16()
        elif self.net == 'resnet50':
            photo_net = self._get_resnet50()
            sketch_net = self._get_resnet50()
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658

        
        print('net')
        print(photo_net)

        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=self.p).cuda()
#         photo_cat_loss = nn.CrossEntropyLoss().cuda()
#         sketch_cat_loss = nn.CrossEntropyLoss().cuda()

#         photo_cat_loss = nn.CrossEntropyLoss()
#         sketch_cat_loss = nn.CrossEntropyLoss()

#         my_triplet_loss = TripletLoss().cuda()
#         my_triplet_loss = TripletLoss().cuda()

        # optimizer
        photo_optimizer = t.optim.SGD(photo_net.parameters(),lr=0.001, momentum=0.9)
        # sketch_optimizer = t.optim.SGD(sketch_net.parameters(),lr=1e-3, 
        #                               weight_decay=0.0005, momentum=0.9)
        # photo_optimizer = t.optim.Adam(photo_net.parameters(), lr=self.lr)
        # sketch_optimizer = t.optim.Adam(sketch_net.parameters(), lr=self.lr)

<<<<<<< HEAD
        exp_lr_scheduler = t.optim.lr_scheduler.StepLR(photo_optimizer, 
                                                       step_size=7, gamma=0.1)

#         triplet_loss_meter = AverageValueMeter()
#         sketch_cat_loss_meter = AverageValueMeter()
#         photo_cat_loss_meter = AverageValueMeter()

        data_loader = TripleDataLoader(self.dataloader_opt)
        dataset = data_loader.load_data()
#         print('Len:', len(dataset))
        loss_list = []
=======
        data_loader = TripleDataLoader(self.dataloader_opt)
        dataset = data_loader.load_data()
        print('Len:', len(dataset))
        for ii, data in enumerate(dataset):
            if ii==2: break
            print(data['L'])
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658

        for epoch in range(self.epochs):

            print('---------------{0}---------------'.format(epoch))

            photo_net.train()
<<<<<<< HEAD
        #     sketch_net.train()
            avg = 0
            count = 0
=======
            sketch_net.train()
            avg_loss = 0
            text = []

>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658
            for ii, data in enumerate(dataset):
                count = ii
                photo_optimizer.zero_grad()
                # sketch_optimizer.zero_grad()

                # photo = .cuda()
                # sketch = data['A'].cuda()
                photo, sketch, neg =Variable(data['P'].cuda()), Variable(data['A'].cuda()), Variable(data['N'].cuda())
#                 photo = data['P']
                photo = photo.unsqueeze(0)
#                 sketch = data['A']
                sketch = sketch.unsqueeze(0)
                neg = neg.unsqueeze(0)
                label = data['L']
                
<<<<<<< HEAD
                s_feature = photo_net(sketch)
                p_feature = photo_net(photo)
                n_feature = photo_net(neg)
#                 s_cat, s_feature = sketch_net(sketch)
=======
                _, a_feature = sketch_net(anchor)
                _, p_feature= photo_net(pos)
                _, n_feature= photo_net(neg)

                # a_feature, p_feature,n_feature = photo_net(anchor, pos, neg)
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658

                # category loss
#                 p_cat_loss = photo_cat_loss(p_cat, label)
#                 s_cat_loss = sketch_cat_loss(s_cat, label)

#                 photo_cat_loss_meter.add(p_cat_loss.item())
#                 sketch_cat_loss_meter.add(s_cat_loss.item())

                # triplet loss
#                 loss = p_cat_loss + s_cat_loss
                loss = triplet_loss(s_feature, p_feature, n_feature)
#                 my_tri_loss = my_triplet_loss(s_feature, p_feature) / (self.batch_size - 1)
#                 triplet_loss_meter.add(my_tri_loss.item())
                # print('my_tri_loss : ', my_tri_loss)

                # print(tri_record - my_tri_loss)
#                 loss = loss + my_tri_loss
                # print('loss :', loss)
                # loss = loss / opt.batch_size
                avg += loss.item()
                loss.backward()

                photo_optimizer.step()
<<<<<<< HEAD
#                 sketch_optimizer.step()
                
                if (ii+1)%64 == 0:
                    print('[Train] Epoch: [{0}][{1}/{2}]\t'
                            'Triplet loss  ({triplet_loss_meterr:.3f})\t'
    #                         'Photo los  ({photo_loss:.4f})\t'
    #                         'Sketch Loss ({sketch_loss:.4f})\t'
                            .format(epoch + 1, ii + 1, 235, 
                                    triplet_loss_meterr=loss.item()))
                
#                 triplet_loss_meter.reset()
#                 photo_cat_loss_meter.reset()
#                 sketch_cat_loss_meter.reset()
                del photo, sketch, neg, s_feature, p_feature, n_feature, loss
                t.cuda.empty_cache()
                
#             if self.save_model:
            t.save(photo_net.state_dict(), self.save_dir + '/photo_' + self.net + '_%s.pth' % epoch)
            # t.save(sketch_net.state_dict(), self.save_dir + '/sketch_' + self.net + '_%s.pth' % epoch)
        loss_list.append(avg/len(dataset))
        print(count)
=======
                sketch_optimizer.step()

                print('[Train] Epoch: [{0}][{1}/{2}]\t'
                        'Triplet loss  ({triplet_loss_meterr:.3f})\t'
                        # 'Loss  ({losss:.4f})\t'
                        # 'Sketch Loss ({acc:.4f})\t'
                        .format(epoch + 1, ii + 1, len(dataset), triplet_loss_meterr=loss.item()))
                avg_loss += loss.item()
                text.append('[Train] Epoch: [{0}][{1}/{2}]\t'
                        'Triplet loss  ({triplet_loss_meterr:.3f})\t'
                        .format(epoch + 1, ii + 1, len(dataset), triplet_loss_meterr=loss.item()))
                if ii==3: break
            if self.save_model:
                t.save(photo_net.state_dict(), self.save_dir + '/photo' + '/photo_' + self.net + '_%s.pth' % epoch)
                t.save(sketch_net.state_dict(), self.save_dir + '/sketch' + '/sketch_' + self.net + '_%s.pth' % epoch)
                # t.save(photo_net, self.save_dir + '/photo' + '/photo_' + self.net + '_%s.pth' % epoch)
                # t.save(sketch_net, self.save_dir + '/sketch' + '/sketch_' + self.net + '_%s.pth' % epoch)
                with open('loss.txt', 'a') as file:
                    # Viết nội dung vào file
                    file.write("---------------{0}---------------\n".format(epoch))
                    for x in text:
                        file.write(x+".\n")
                    file.write("Train loss: " + str(avg_loss/len(dataset)) + "\n")

>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658







