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
weights=torch.load("/kaggle/input/notebooks/ziadabdallah047/baseline4/B4_bestmodel.pth")
model.load_state_dict(weights)
device=("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count()>1:
    model=nn.DataParallel(model)
model.to(device)

test_videos = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]
testset=B4Dataset(test_videos)
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