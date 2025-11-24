import os,argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models.Padiff_arch.MLWNet_arch import MLWNet
from metrics import *
from skimage.metrics import structural_similarity
import torch
import torchvision
from pytorch_model_summary import summary
import torchvision
from ptflops import get_model_complexity_info

import os,argparse

#import lpips
import numpy as np
from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models.Padiff_arch.MLWNet_arch import MLWNet
from models.CDUnetv2 import CDUnetV2
from models.LGNet import LGNet
from metrics import *

from skimage.metrics import structural_similarity
abs=os.getcwd()+'/'
def tensorShow(tensors,titles):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='ots',help='its or ots')
parser.add_argument('--test_imgs',type=str,default='test_ufo',help='Test imgs folder')
opt=parser.parse_args()
dataset=opt.task
haze_dir = r'D:\low_light\dataset\lowtest\\'
#clear_dir = r'D:\low_light\dataset\v2\Test\Normal'
model_dir = r"D:\low_light\Low_net1\trained_models\mfediff\LoLv1_p23.735s0.847m.pt"
output_dir = r'D:\low_light\dataset\lowout\\'
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
#loss_fn = lpips.LPIPS(net='alex',version=0.1)
device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
net=MLWNet()
net=nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()
psnr_list = []
ssim_list = []
lp_list = []
for im in os.listdir(haze_dir):
    haze = Image.open(os.path.join(haze_dir, im)).convert('RGB')
    imc = im.split('/')[-1].split('low')[-1]  #修改
    #clear = Image.open(os.path.join(clear_dir, 'normal'+imc)).convert('RGB')
    haze1 = tfs.Compose([
        #tfs.Resize((256,256)),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(haze)[None, ::]
    haze1 = haze1.to(device)
    #clear_no = tfs.ToTensor()(clear)[None, ::]

    with torch.no_grad():
        pred, _, _, _ = net(haze1) #修改
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    #pp = psnr(pred.cpu(), clear_no)
    #ss = ssim(pred.cpu(), clear_no)
   # lp = loss_fn.forward(pred.cpu(), clear_no)
    #psnr_list.append(pp)
    #ssim_list.append(ss)
#    lp_list.append(lp)
    vutils.save_image(ts, os.path.join(output_dir, im))


#print(f'Average PSNR is {np.mean(psnr_list)}')
#print(f'Average SSIM is {np.mean(ssim_list)}')
#print(f'Average LPIPS is {torch.mean(torch.stack(lp_list))}')



