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

class B1DataSet(data.Dataset):
    def __init__(self,paths,train=True):
        super(B1DataSet,self).__init__()
        self.all_paths=[]
        self.train=train
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
        image=Image.open(self.all_paths[index]).convert("RGB")
        image_name=os.path.basename(self.all_paths[index])
        label=self.labels[image_name]
        if self.train:
            if self.transformtrain:
                image=self.transformtrain(image)
        elif self.train==False:
            if self.transformval:
                image=self.transformval(image)

        return image,label


train_videos = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
val_videos = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]


b1traindata=B1DataSet(train_videos,train=True)
b1valdata=B1DataSet(val_videos,train=False)

trainloader=data.DataLoader(b1traindata,batch_size=64,shuffle=True)
valloader=data.DataLoader(b1valdata,batch_size=64)

device=("cuda" if torch.cuda.is_available() else "cpu")
model=torchvision.models.resnet50(weights="DEFAULT")
layers=list(model.children())[:-1]
layers.extend([nn.Flatten(),
                nn.Dropout()
               ,nn.Linear(2048,1024),
               nn.ReLU(inplace=True),
               nn.Dropout(),
               nn.Linear(1024,512),
               nn.ReLU(inplace=True),
               nn.Linear(512,8)])
model=nn.Sequential(*layers)
if torch.cuda.device_count()>1:
    model=nn.DataParallel(model)
model.to(device)
optimizier=optim.AdamW(model.parameters(),lr=.0001)
criterion=nn.CrossEntropyLoss()
epochs=20

valaccuracy=[]
trainaccuracy=[]
last3models=[]
bestaccval=0.0
for epoch in range(epochs):
    totaltrain,correcttrain,totalval,correctval=0,0,0,0
    epoch_loss=0.0
    i=0
    model.train()
    for inputs,labels in trainloader:
        i+=1
        print(f"batch {i} / {len(trainloader)}")
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
        torch.save(model.state_dict(),"B1_bestmodel.pth")
    checkpointname=f"B1_model_{epoch}.pth"
    last3models.append(checkpointname)
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
