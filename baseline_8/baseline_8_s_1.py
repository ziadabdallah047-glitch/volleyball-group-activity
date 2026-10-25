import random
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
       
        self.weight = weight 

        self.gamma = gamma   
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class B5Dataset(data.Dataset):
    def __init__(self,paths,train=True):
        super(B5Dataset,self).__init__()
        self.train=train
        self.all_paths=[]
        self.transformtrain=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.RandomRotation(degrees=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.transformval=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.class_to_idx = {"standing":0,
                     "blocking":1,
                     "moving":2,
                     "spiking":3,
                     "falling":4,
                     "digging":5,
                     "waiting":6,
                     "setting":7,
                     "jumping":8
                                
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

                        
                        
    def __len__(self):
        return len(self.all_paths)


    def __getitem__(self, index):
        
        players=[[],[],[],[],[],[],[],[],[],[],[],[]]
        path,annot=self.all_paths[index]
        labels=[]
        masks=[]
        image_number=int(path.split("/")[1])
        i=0
        for i_n in range (image_number-4,image_number+5):
            image_path=os.path.join("/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",path,f"{i_n}.jpg")
            with Image.open(image_path) as image:
                for p in range(12):
                    p_annot=annot[p*20+6+i].split()
                    p_dim=(int(p_annot[1]),int(p_annot[2]),int(p_annot[3]),int(p_annot[4]))
                    p_l=self.class_to_idx[p_annot[-1]]
                    if i ==4:
                        labels.append(p_l)
                        if self.train==True and p_l==0:
                            if random.random()<0.1:
                                masks.append(1.0)
                            else:
                                masks.append(0.0)
                        else:
                            masks.append(1.0)
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
        label_tensor=torch.tensor(labels,dtype=torch.long)
        mask_tensor=torch.tensor(masks,dtype=torch.float32)
        return image_stacked,label_tensor,mask_tensor

class FeatureExtractionTemporal(nn.Module):
    def __init__(self,backbone):
        super(FeatureExtractionTemporal, self).__init__()
        self.backbone=backbone
        self.lstm=nn.LSTM(input_size=2048,hidden_size=512,num_layers=2,batch_first=True,bidirectional=True)
        self.classifier=nn.Sequential(nn.Linear(1024,512),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.3),
                                      nn.Linear(512,256),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.3),
                                      nn.Linear(256,9))

    def forward(self,x):
        B,P,F,C,H,W=x.size()
        x=x.view(B*P*F,C,H,W)
        x=self.backbone(x)
        x=x.view(B*P,F,2048)
        output,(h_n,c_n)=self.lstm(x)
        x=self.classifier(output[:,4,:])
        return x

train_videos = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
val_videos = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]


traindata=B5Dataset(train_videos)
valdata=B5Dataset(val_videos,train=False)

trainloader=data.DataLoader(traindata,batch_size=1,shuffle=True,num_workers=4)
valloader=data.DataLoader(valdata,batch_size=2,num_workers=4)

device=("cuda" if torch.cuda.is_available() else "cpu")

backbone=torchvision.models.resnet50(weights="DEFAULT")
layers=list(backbone.children())[:-1]

backbone=nn.Sequential(*layers)
model=FeatureExtractionTemporal(backbone)
weights=torch.load("/kaggle/input/notebooks/ziadabdallah47/newbaseline5/B5_s1_model_8.pth")
model.load_state_dict(weights)
if torch.cuda.device_count()>1:
    model=nn.DataParallel(model)
model.to(device)

optimizier = optim.AdamW(model.parameters(),lr=.0001, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizier, mode='max', factor=0.1, patience=2)
criterion = FocalLoss(reduction='none')

epochs=4

valaccuracy=[]
trainaccuracy=[]
last3models=[]
bestaccval=0.0

for epoch in range(epochs):
    totaltrain,correcttrain,totalval,correctval=0,0,0,0
    epoch_loss=0.0
    
    model.train()
    for i,(inputs,labels,masks) in enumerate(trainloader):
        
        if i%50==0:
            print(f"batch {i} / {len(trainloader)} epoch {epoch}")
        labels=labels.view(-1)
        masks = masks.view(-1).to(device)
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        _,predicted=torch.max(outputs,1)
        correct_preds = (predicted == labels) * masks 
        correcttrain += correct_preds.sum().item()
        totaltrain += masks.sum().item()
        
        loss=criterion(outputs,labels)
        loss = loss * masks
        loss = loss.sum() / masks.sum().clamp(min=1e-8)
        loss=loss/16
        epoch_loss+=loss.item()*16
        loss.backward()
        if(i+1)%16==0:
            optimizier.step()
            optimizier.zero_grad()
    if len(trainloader)%16!=0:
        optimizier.step()
        optimizier.zero_grad()
        
    trainaccuracy.append(correcttrain/totaltrain)
    print(f"epoch loss {epoch_loss/len(trainloader)}")
    model.eval()
    with torch.no_grad():
        for inputs,labels ,_ in valloader:
            labels = labels.view(-1)
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            _,predicted=torch.max(outputs,1)
            correctval+=(predicted==labels).sum().item()
            totalval+=labels.size(0)
        accval=correctval/totalval
        valaccuracy.append(accval)
        scheduler.step(accval)
    if accval>bestaccval:
        bestaccval=accval
        if isinstance(model,nn.DataParallel):
            torch.save(model.module.state_dict(),"B5_s1_bestmodel.pth")
        else:
            torch.save(model.state_dict(),"B5_s1_bestmodel.pth")
    checkpointname=f"B5_s1_model_{epoch}.pth"
    last3models.append(checkpointname)
    if isinstance(model,nn.DataParallel):
        torch.save(model.module.state_dict(),checkpointname)
    else:
        torch.save(model.state_dict(),checkpointname)
    if len(last3models)>3:
        oldestmodel=last3models.pop(0)
        if os.path.exists(oldestmodel):
            os.remove(oldestmodel)
    clear_output(wait=True)
    plt.figure(figsize=(15,10))
    plt.plot(trainaccuracy,label="train accuracy",color="blue",marker='o')
    plt.plot(valaccuracy,label="validation accuracy",color="red",marker="s")
    plt.title("train accuracy vs validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.show()
    print(f"validation accuracy {valaccuracy[-1]:.4f} , train accuracy {trainaccuracy[-1]:.4f} ")
    plt.close("all")