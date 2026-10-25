import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision
import os
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim as optim
from IPython.display import clear_output

class B7s2Dataset(data.Dataset):
    def __init__(self,paths,train=True):
        super(B7s2Dataset,self).__init__()
        self.train=train
        self.labels={}
        self.all_paths=[]
        self.transformtrain=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.transformval=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.class_to_idx = {'l_set': 0,
                                        'r_set': 1,
                                        'l-pass': 2,
                                        'r-pass': 3,
                                        'l-spike': 4,
                                        'r_spike': 5,
                                        'l_winpoint': 6,
                                        'r_winpoint': 7}
                            
        for b_f_n in paths:
            b_folder=os.path.join("/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",f"{b_f_n}")
            for folder in os.listdir(b_folder):
                file=os.path.join(b_folder,folder,f"{folder}.txt")
                with open(file,"r") as f:
                    annotation=f.read().splitlines()
                    if len(annotation)==240:
                        path=os.path.join(b_f_n,folder)
                        self.all_paths.append((path,annotation))
        for b_folder in paths:
        
                    with open(os.path.join("/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",b_folder,"annotations.txt"),"r")as f:
                        for line in f:
                            label=line.split()[:2]
                            self.labels[label[0]]=self.class_to_idx[label[1]]
    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, index):
        players=[[],[],[],[],[],[],[],[],[],[],[],[]]
        path,annot=self.all_paths[index]
        image_number=int(path.split("/")[1])
        label=self.labels[f"{image_number}.jpg"]
        i=0
        for i_n in range (image_number-4,image_number+5):
            image_path=os.path.join("/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",path,f"{i_n}.jpg")
            with Image.open(image_path).convert("RGB") as image:
                for p in range(12):
                    p_annot=annot[p*20+6+i].split()
                    p_dim=(int(p_annot[1]),int(p_annot[2]),int(p_annot[3]),int(p_annot[4]))
                    p_i=image.crop(p_dim)
                    if self.train:
                        if self.transformtrain:
                            p_i=self.transformtrain(p_i)
                    elif self.train==False:
                        if self.transformval:
                            p_i=self.transformval(p_i)
                    
                    players[p].append(p_i)
        
            i+=1
        image_stacked=torch.stack([torch.stack(player) for player in players])
        
        return image_stacked,label
class FeatureExtractionTemporal(nn.Module):
    def __init__(self,backbone):
        super(FeatureExtractionTemporal, self).__init__()
        self.backbone=backbone
        self.lstm=nn.LSTM(input_size=2048,hidden_size=512,num_layers=2,batch_first=True,bidirectional=True)

    def forward(self,x):
        B,P,F,C,H,W=x.size()
        x=x.view(B*P*F,C,H,W)
        x=self.backbone(x)
        x = x.squeeze(-1).squeeze(-1)
        x=x.view(B*P,F,2048)
        output,(h_n,c_n)=self.lstm(x)
        lstm_output=output.contiguous().view(B,P,F,1024)
        featur_extracted=x=x.view(B,P,F,2048)
        x=torch.cat((featur_extracted,lstm_output),dim=-1)
        return x
class B7network(nn.Module):
    def __init__(self, ):
        super(B7network,self).__init__()
        self.lstm=nn.LSTM(input_size=3072,hidden_size=512,num_layers=2,batch_first=True,bidirectional=True)
        self.classifier=nn.Sequential(nn.Linear(1024, 512), 
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(256, 8))
    def forward(self,x):
        x,_=torch.max(x,1)
        output,(h_n,c_n)=self.lstm(x)
        x=output[:,4,:]
        x=self.classifier(x)
        return x

device=("cuda" if torch.cuda.is_available() else "cpu")
train_videos = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
val_videos = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
batch_size=2
trainset=B7s2Dataset(train_videos)
trainloader=data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=4)
valset=B7s2Dataset(val_videos,train=False)
valloader=data.DataLoader(valset,batch_size=batch_size,num_workers=4)
backbone=torchvision.models.resnet50(weights="DEFAULT")
layers=list(backbone.children())[:-1]
backbone=nn.Sequential(*layers)
model1=FeatureExtractionTemporal(backbone)
weights=torch.load("/kaggle/input/notebooks/ziadabdallah47/baseline-5/B5_s1_model_3.pth")
model1.load_state_dict(weights,strict=False)
for param in model1.parameters():
    param.requires_grad=False
if torch.cuda.device_count()>1:
    model1=nn.DataParallel(model1)
model1.to(device)
model=B7network()

weights=torch.load("/kaggle/input/datasets/ziadabdallah47/base7v1/B7_s2_bestmodel.pth")
model.load_state_dict(weights)
if torch.cuda.device_count()>1:
    model=nn.DataParallel(model)
model.to(device)

test_videos = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]
testset=B7s2Dataset(test_videos,train=False)
testloader=data.DataLoader(testset,batch_size=3,num_workers=4)


classes=['l_set','r_set','l-pass','r-pass','l-spike','r_spike','l_winpoint','r_winpoint']

all_preds = []
all_labels = []
        
correct,total=0,0

model.eval()

with torch.no_grad():
    for inputs, labels in testloader:
       
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        
        feture_extracted=model1(inputs)
        outputs=model(feture_extracted) 
        
        _, predicted = torch.max(outputs, 1)
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)
       
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


cm = confusion_matrix(all_labels, all_preds)


plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()
print(f"acurracy :{correct/total}")
