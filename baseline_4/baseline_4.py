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

class B4Dataset(data.Dataset):
    def __init__(self,paths):
        super(B4Dataset,self).__init__()
        self.all_paths=[]
        self.transform=transforms.Compose([
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
        self.labels={}
        for b_folder in paths:
            b_folder=os.path.join("/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",b_folder)
            for folder in os.listdir(b_folder):
                frame_folder=os.path.join(b_folder,folder)
                if os.path.isdir(frame_folder):
                    self.all_paths.append(os.path.join(frame_folder,folder+".jpg"))
        for b_folder in paths:

            with open(os.path.join("/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",b_folder,"annotations.txt"),"r")as f:
                for line in f:
                    label=line.split()[:2]
                    self.labels[label[0]]=self.class_to_idx[label[1]]
    def __len__(self):
        return len(self.all_paths)


    def __getitem__(self, index):
        images=[]
        image_name=os.path.basename(self.all_paths[index])
        label=self.labels[image_name]
        image_number=int(image_name[:-4])
        dirname=os.path.dirname(self.all_paths[index])
        for number in range(image_number-4,image_number+5):
            
            path=os.path.join(dirname,f"{number}.jpg")
            image=Image.open(path).convert("RGB")
            if self.transform:
                image=self.transform(image)
            images.append(image)
        staked_images=torch.stack(images)
        

        return staked_images,label


class TemporalImageLevel(nn.Module):
    def __init__(self,backbone):
        super(TemporalImageLevel,self).__init__()
        self.backbone=backbone
        self.lstm=nn.LSTM(input_size=2048,hidden_size=512,num_layers=2,batch_first=True,bidirectional=True)
        self.classifier=nn.Sequential(nn.Linear(1024,512),
                                      nn.ReLU(),
                                      nn.Linear(512,8))

    def forward(self,x):
            B,f,c,h,w=x.size()
            x=x.view(B*f,c,h,w)
            x=self.backbone(x)
            x=x.view(-1,f,2048)
            output,(h_n,c_n)=self.lstm(x)
            x=output[:,4,:]
            x=self.classifier(x)
            return x

backbone=torchvision.models.resnet50(weights="DEFAULT")
features=list(backbone.children())[:-1]
backbone=nn.Sequential(*features)

model=TemporalImageLevel(backbone)

train_videos = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
val_videos = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]

b4traindata=B4Dataset(train_videos)
b4valdata=B4Dataset(val_videos)

trainloader=data.DataLoader(b4traindata,batch_size=8,shuffle=True)
valloader=data.DataLoader(b4valdata,batch_size=8)

device=("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count()>1:
    model=nn.DataParallel(model)
model.to(device)
optimizier=optim.AdamW(model.parameters(),lr=.0001)
criterion=nn.CrossEntropyLoss()
epochs=10

valaccuracy=[]
trainaccuracy=[]
last3models=[]
bestaccval=0.0
for epoch in range(epochs):
    totaltrain,correcttrain,totalval,correctval=0,0,0,0
    epoch_loss=0.0
    batch=0
    model.train()
    for inputs,labels in trainloader:
        batch+=1
        if batch%50==0:
            print(f"batch number {batch} / {len(trainloader)} epoch number {epoch}")
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        _,predicted=torch.max(outputs,1)
        correcttrain+=(predicted==labels).sum().item()
        totaltrain+=labels.size(0)
        optimizier.zero_grad()
        loss=criterion(outputs,labels)
        epoch_loss+=loss.item()
        loss.backward()
        optimizier.step()
    trainaccuracy.append(correcttrain/totaltrain)
    print(f"epoch loss {epoch_loss/len(trainloader)}")
    model.eval()
    with torch.no_grad():
        for inputs,labels in valloader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            _,predicted=torch.max(outputs,1)
            correctval+=(predicted==labels).sum().item()
            totalval+=labels.size(0)
        accval=correctval/totalval
        valaccuracy.append(accval)
    if accval>bestaccval:
        bestaccval=accval
        if isinstance(model,nn.DataParallel):
            torch.save(model.module.state_dict(),"B4_bestmodel.pth")
        else:
            torch.save(model.state_dict(),"B4_bestmodel.pth")
    checkpointname=f"B4_model_{epoch}.pth"
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
    print(f"validation accuracy {valaccuracy[epoch]:.4f} , train accuracy {trainaccuracy[epoch]:.4f} ")

