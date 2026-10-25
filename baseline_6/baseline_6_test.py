import torch
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
import matplotlib.pyplot as plt

class B6s2Dataset(data.Dataset):
    def __init__(self,paths,train=True):
        super(B6s2Dataset,self).__init__()
        self.train=train
        self.labels={}
        self.all_paths=[]
        self.transformtrain=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.transformval=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.class_to_idx = {
                                        'l_set': 0,
                                        'r_set': 1,
                                        'l-pass': 2,
                                        'r-pass': 3,
                                        'l-spike': 4,
                                        'r_spike': 5,
                                        'l_winpoint': 6,
                                        'r_winpoint': 7
                                    }
                            
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

class B6network(nn.Module):
    def __init__(self,backbone):
        super(B6network,self).__init__()
        self.backbone=backbone
        self.lstm=nn.LSTM(input_size=2048,hidden_size=512,num_layers=2,batch_first=True,bidirectional=True)
        self.classifier=nn.Sequential(nn.Linear(1024,512),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(512,8))
    def train(self,mode=True):
        super(B6network,self).train(mode)
        self.backbone.eval()
        return self
    def forward(self,x):
        B,P,F,C,H,W=x.size()
        x=x.view(B*P*F,C,H,W)
        x=self.backbone(x)
        x=x.view(B,P,F,2048)
        x,_=torch.max(x,1)
        output,(h_n,c_n)=self.lstm(x)
        x=output[:,4,:]
        x=self.classifier(x)
        return x

    


train_videos = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
val_videos = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
batch_size=2
device=("cuda" if torch.cuda.is_available() else "cpu")

trainset=B6s2Dataset(train_videos)
trainloader=data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
valset=B6s2Dataset(val_videos,train=False)
valloader=data.DataLoader(valset,batch_size=batch_size)

backbone=torchvision.models.resnet50(weights="DEFAULT")

layers=list(backbone.children())[:-1]

layers.extend([nn.Flatten(),
               nn.Linear(2048,1024),
               nn.ReLU(inplace=True),
               nn.Dropout(),
               nn.Linear(1024,9)])
backbone=nn.Sequential(*layers)

weights=torch.load("/kaggle/input/datasets/ziadabdallah047/vollyball-baseline3-s1-model/B3_s1_bestmodel.pth")

backbone.load_state_dict(weights)
feature=list(backbone.children())[:-5]
backbone=nn.Sequential(*feature)
for param in backbone.parameters():
    param.requires_grad=False
backbone.eval()

model=B6network(backbone)
weights=torch.load("/kaggle/input/notebooks/ziadabdallah047/baseline-6/B6_s2_bestmodel.pth")
model.load_state_dict(weights)
if torch.cuda.device_count()>1:
    model=nn.DataParallel(model)
model.to(device)
test_videos = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]
testset=B6s2Dataset(test_videos)
testloader=data.DataLoader(testset,batch_size=8)
correct=0
total=0

model.eval()
with torch.no_grad():
    for inputs,labels in testloader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        _,predicted=torch.max(outputs,1)
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)
    print(f"acurracy :{correct/total}")

