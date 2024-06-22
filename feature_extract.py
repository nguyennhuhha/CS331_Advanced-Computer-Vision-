# from data import TripleDataLoader
from sklearn.utils import resample
from utils.extractor import Extractor
from models.vgg import vgg16
# from models.sketch_resnet import resnet50
import torch as t
from torch import nn
import os

# The script to extract sketches or photos' features using the trained model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
train_set_root = 'dataset/sketch_train'
test_set_root = 'dataset/sketch_test'

train_photo_root = 'dataset/photo_train'
test_photo_root = 'dataset/photo_test'
for i in range(0,30):
    print('---------------',i,'---------------')
    # The trained model root for resnet
    # model at epoch 85th
    SKETCH_RESNET = 'model/rn50_bs32_mg1_lr3_10class/photo_resnet50_'+str(i)+'.pth' #change this
    PHOTO_RESNET = 'model/rn50_bs32_mg1_lr3_10class/photo_resnet50_'+str(i)+'.pth' #change this

    # The trained model root for vgg
    # SKETCH_VGG = 'model/vgg/sketch/sketch_vgg16_0.pth'
    # PHOTO_VGG = 'model/vgg/photo/photo_vgg16_0.pth'

<<<<<<< HEAD
    # FINE_TUNE_RESNET = '/data1/zzl/model/caffe2torch/fine_tune/model_270.pth'

    device = 'cuda:1'
=======
# The trained model root for vgg
# SKETCH_VGG = 'model/sketch_vgg16_401.pth'
PHOTO_VGG = 'model/bt32_001_1/photo_vgg16_19.pth'
a = PHOTO_VGG.split('_')
epoch =''+ a[-1].split('.')[0]
# FINE_TUNE_RESNET = '/data1/zzl/model/caffe2torch/fine_tune/model_270.pth'
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658

    # '''vgg'''
    # vgg = t.load(PHOTO_VGG)
    # # vgg.classifier[0] = nn.Linear(in_features=512*7*7, out_features=4096, bias=True)
    # # vgg.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
    # # vgg.load_state_dict(t.load(PHOTO_VGG, map_location=t.device('cpu')))
    # # vgg.cuda()

<<<<<<< HEAD
    # ext = Extractor(vgg)
    # ext.reload_model(vgg)
    # vgg.eval()

    # photo_feature = ext._extract_with_dataloader(test_photo_root, 'photo-vgg-0epoch.pkl')

    # vgg.load_state_dict(t.load(SKETCH_VGG, map_location=t.device('cpu')))
    # ext.reload_model(vgg)

    # sketch_feature = ext._extract_with_dataloader(test_set_root, 'sketch-vgg-0epoch.pkl')

    '''resnet'''
    resnet = resnet50()
    resnet.fc = nn.Linear(in_features=2048, out_features=10)

    # resnet.load_state_dict(t.load(PHOTO_RESNET, map_location=t.device('cpu')))
    # resnet.to(device)
=======
'''vgg'''
vgg = vgg16(pretrained=False)
vgg.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
# print(vgg)
# vgg = t.load(PHOTO_VGG, map_location=t.device('cpu'))
# vgg.load_state_dict(checkpoint)
# vgg.classifier[0] = nn.Linear(in_features=512*7*7, out_features=4096, bias=True)
# vgg.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
vgg.load_state_dict(t.load(PHOTO_VGG, map_location=t.device('cpu')))
# vgg.cuda()

ext = Extractor(vgg)
ext.reload_model(vgg)
# vgg.eval()

photo_feature = ext._extract_with_dataloader(test_photo_root, 'feature/bt32_001_1/photo-vgg' + '-%sepoch.pkl'%epoch)
# vgg= t.load(SKETCH_VGG)
# ext.reload_model(vgg)

sketch_feature = ext._extract_with_dataloader(test_set_root, 'feature/bt32_001_1/sketch-vgg' + '-%sepoch.pkl'%epoch)

'''resnet'''

# resnet = resnet50()
# resnet.fc = nn.Linear(in_features=2048, out_features=125)
# resnet.load_state_dict(t.load(PHOTO_RESNET, map_location=t.device('cpu')))
# # resnet.cuda()
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658

    if t.cuda.is_available():
            device = t.device("cuda") 
            resnet.to(device)  # Move model to GPU
    else:
        device = t.device("cpu")
        print("CUDA is not available. Running on CPU.")

    ext = Extractor(e_model=resnet)
    ext.reload_model(resnet)
    resnet.eval()

    # photo_feature = ext._extract_with_dataloader(test_photo_root, 'features_pkl/rn50_bs32_mg1_lr3_10class/'+str(i)+'/photo-resnet-epoch_'+str(i)+'.pkl') # change this 2 places

    resnet.load_state_dict(t.load(SKETCH_RESNET, map_location=t.device('cpu')))
    ext.reload_model(resnet)

    sketch_feature = ext._extract_with_dataloader(test_set_root, 'features_pkl/rn50_bs32_mg1_lr3_10class/'+str(i)+'/sketch-resnet-epoch_'+str(i)+'.pkl') # change this 2 places
    print('Done: ',i)
# python feature_extract.py