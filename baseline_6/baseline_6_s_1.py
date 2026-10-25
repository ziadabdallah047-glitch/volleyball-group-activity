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

class B3DataSet(data.Dataset):
    def __init__(self,path):
        super(B3DataSet,self).__init__()
        paths={}
        self.all_paths=[]
        self.transform=transforms.Compose([
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
        
        counter=0
        for b_folder in path:
            b_folder=os.path.join("/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",b_folder)
            for folder in os.listdir(b_folder):
                frame_folder=os.path.join(b_folder,folder)
                if os.path.isdir(frame_folder):
                    paths[folder+".jpg"]=os.path.join(frame_folder,folder+".jpg")
        for b_folder in path:

            with open(os.path.join("/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",b_folder,"annotations.txt"),"r")as f:
                for line in f:
                    label=line.split()[:]
                    for i in range(1,len(label)-1,5):
                        if counter >=10000 and label[i+3]=="standing":
                            continue
                        if label[i+3]=="standing":
                            counter+=1
                        x1=int(label[i+1])
                        y1=int(label[i+2])
                        x2=x1+int(label[i+3])
                        y2=y1+int(label[i+4])
                        dim=(x1,y1,x2,y2)
                        
                        self.all_paths.append([paths[label[0]],dim,self.class_to_idx[label[i+5]]])
    def __len__(self):
        return len(self.all_paths)


    def __getitem__(self, index):
        image=Image.open(self.all_paths[index][0]).convert("RGB")
        
        image=image.crop(self.all_paths[index][1])
        label=self.all_paths[index][2]
        if self.transform:
            image=self.transform(image)

        return image,label


train_videos = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
val_videos = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]

test_videos = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]
b1traindata=B3DataSet(train_videos)
b1valdata=B3DataSet(val_videos)

trainloader=data.DataLoader(b1traindata,batch_size=48,shuffle=True)
valloader=data.DataLoader(b1valdata,batch_size=48)

device=("cuda" if torch.cuda.is_available() else "cpu")

model=torchvision.models.resnet50(weights="DEFAULT")
layers=list(model.children())[:-1]
layers.extend([nn.Flatten(),
                nn.Dropout()
               ,nn.Linear(2048,1024),
               nn.ReLU(inplace=True),
               nn.Linear(1024,9)])
model=nn.Sequential(*layers)
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
    
    model.train()
    for inputs,labels in trainloader:
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
        torch.save(model.state_dict(),"B3bestmodel.pth")
    checkpointname=f"B3model_{epoch}.pth"
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
