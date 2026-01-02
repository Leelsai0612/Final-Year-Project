test=r"D:\My_Project\IIIT5K-Word_V3.0\IIIT5K\test"
train=r"D:\My_Project\IIIT5K-Word_V3.0\IIIT5K\train"
print("Done!")

import os
import time
import sys
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/clovaai/deep-text-recognition-benchmark.git'
project_name = splitext(basename(git_repo_url))[0]

# Check if the project directory already exists
if not exists(project_name):
    # Clone the Git repository
    os.system(f'git clone  {git_repo_url}')
  
# Add the project directory to the system path
sys.path.append(project_name)
print("Done!")

# Downloading Pretrained models
import os
import gdown

pretrained_model_path = "./pretrained_model/"

# Create the directory if it doesn't exist
if not os.path.exists(pretrained_model_path):
    os.makedirs(pretrained_model_path)

# List of tuples containing the filename and file ID from Google Drive
links = [
    # Best Accuracy, Note you need to modify model architecture for it
    ("TPS-ResNet-BiLSTM-Attn.pth", "1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9"),
    # Best For Case Sensitive
    ("TPS-ResNet-BiLSTM-Attn-case-sensitive.pth", "1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY")
]

# Loop over each item in the links list
for counter, item in enumerate(links):
    filename = item[0]
    fileid = item[1]

    # Generate the download link using the file ID
    download_link = f"https://drive.google.com/uc?id={fileid}"
    # Set the output file path
    file_path = os.path.join(pretrained_model_path, filename)

    # Download the file using gdown
    gdown.download(download_link, file_path, quiet=False)

# Importing deps
import os
import time
import string
import argparse

#import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model
print("Done!")

# Defining Model Architecture
#!export CUDA_VISIBLE_DEVICES=0
parser = argparse.Namespace()
parser.eval_data=True
parser.benchmark_all_eval='store_true'
parser.workers=4
parser.batch_size=192
parser.saved_model='./pretrained_model/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth'
# parser.saved_model='./pretrained_model/TPS-ResNet-BiLSTM-Attn.pth'

""" Data processing """
parser.batch_max_length=25
parser.imgH=32
parser.imgW=100
parser.rgb='store_true'
parser.character='0123456789abcdefghijklmnopqrstuvwxyz'
parser.sensitive='store_true'
parser.PAD='store_true'

""" Model Architecture """
parser.Transformation='TPS'
parser.FeatureExtraction='ResNet'
parser.SequenceModeling='BiLSTM'
parser.Prediction='Attn'
parser.num_fiducial=20
parser.input_channel=1
parser.output_channel=512
parser.hidden_size=256
parser.num_gpu=0

opt = parser
print(opt)

""" vocab / character number configuration """
opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()

print(opt.character)

# If CTC is present we will use CTC otherwise Attention label Converter
if 'CTC' in opt.Prediction:
    converter = CTCLabelConverter(opt.character)
else:
    converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)

print("Done!")

# Initializing Model
opt.input_channel = 1
model = Model(opt)
print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
      opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
      opt.SequenceModeling, opt.Prediction)
try:
  model = torch.nn.DataParallel(model).cuda()
except:
  model = torch.nn.DataParallel(model)  # Remove .cuda()
  map_location=torch.device('cpu')

print('loading pretrained model from %s' % opt.saved_model)

# Loading the model
print('loading pretrained model from %s' % opt.saved_model)
try:
  model.load_state_dict(torch.load(opt.saved_model))
except:
  model.load_state_dict(torch.load(opt.saved_model, map_location=torch.device('cpu')))
opt.experiment_name = '_'.join(opt.saved_model.split('/')[1:])

os.makedirs(f'./result/{opt.experiment_name}', exist_ok=True)
os.system(f'cp {opt.saved_model} ./result/{opt.experiment_name}/')

# Defining loss
try:
  # For CUDA
  criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda() 
except:
  criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
# model.eval()
print("Done!")

# Chosing and displaying test images
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


#%matplotlib inline

test_images = [
    (test + "/1002_1.png", "PRIVATE" ),
    (test + "/1002_2.png", "PARKING" ),
    (test + "/1009_1.png", "Salutes" ),
    (test + "/100_1.png", "DOLCE" ),
    (test + "/100_2.png", "GABBANA" ),
    (test + "/27_4.png", "India" ),
    (test + "/69_2.png", "LIKE" ),
]

input_datas = []

n_images = len(test_images)
fig = plt.figure(figsize=(1,4))
fig.suptitle("")
for n, (imagepath, label) in enumerate(test_images):
    # a = fig.add_subplot(np.ceil(n_images/float(1)), 1, n + 1)
  a = fig.add_subplot(int(np.ceil(n_images/float(1))), 1, n + 1)
  image = Image.open(imagepath).convert('L')
  plt.imshow(image, cmap="Purples")
  a.set_title(label)
  input_datas.append((image, label))
fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
plt.show()

from dataset import AlignCollate
_AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
image_tensors, labels = _AlignCollate(input_datas)

print(labels)

evaluation_loader = torch.utils.data.DataLoader(
            input_datas, batch_size=2,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=_AlignCollate, pin_memory=True)
print("Created evaluation loader")

# Predicting the labels of the chosen images
import torch

max_length = 10
total_correct = 0
total_samples = 0

with torch.no_grad():
    for i, (cpu_images, cpu_texts) in enumerate(evaluation_loader):
        batch_size = cpu_images.size(0)
        length_of_data = 0 + batch_size
        
        # Move data to GPU if available
        if torch.cuda.is_available():
            image = cpu_images.cuda()
            length_for_pred = torch.cuda.IntTensor([max_length] * batch_size)
            text_for_pred = torch.cuda.LongTensor(batch_size, max_length + 1).fill_(0)
            text_for_loss, length_for_loss = converter.encode(cpu_texts)
        else:
            image = cpu_images
            length_for_pred = torch.IntTensor([max_length] * batch_size)
            text_for_pred = torch.LongTensor(batch_size, max_length + 1).fill_(0)
            text_for_loss, length_for_loss = converter.encode(cpu_texts)
        preds = model(image, text_for_pred, is_train=False)
        preds = preds[:, :text_for_loss.shape[1] - 1, :]
        target = text_for_loss[:, 1:]
        _, preds_index = preds.max(2)
        
        sim_preds = converter.decode(preds_index, length_for_pred)
        sim_preds = list(map(lambda s: s.replace("[s]", "").replace("[GO]", ""), sim_preds))
        
        # Calculate accuracy
        correct = sum([1 for gt, pred in zip(cpu_texts, sim_preds) if gt == pred])
        total_correct += correct
        total_samples += batch_size
        
        print("Exact labels:", cpu_texts)
        print("Predicted   :", sim_preds)
    
    accuracy = total_correct / total_samples
    print("Accuracy    :", accuracy)


    