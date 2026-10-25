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
from collections import OrderedDict
train_videos = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
val_videos = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class B3Dataset(data.Dataset):
    def __init__(self,paths):
        super(B3Dataset,self).__init__()
        self.all_paths=[]
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
        self.transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        
          
        self.annotiation={}
        for b_folder in paths:
            with open(os.path.join("/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",b_folder,"annotations.txt"),"r")as f:
                for line in f:
                    label=line.split()[:]
                    if int((len(label)-2)/5)>10:
                        self.annotiation[label[0]]=label[1:]
        for b_folder in paths:
            b_folder=os.path.join("/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",b_folder)
            for folder in os.listdir(b_folder):
                frame_folder=os.path.join(b_folder,folder)
                if os.path.isdir(frame_folder):
                    image_name=f"{folder}.jpg"
                    if image_name in self.annotiation:
                        self.all_paths.append(os.path.join(frame_folder,image_name))
        

    def __len__(self):
            return len(self.all_paths)
    
    
    def __getitem__(self, index):
            image=Image.open(self.all_paths[index]).convert("RGB")
            image_name=os.path.basename(self.all_paths[index])
            label=self.annotiation[image_name]
            images=[]
            T_label=self.class_to_idx[label[0]]
            
            for i in range(1,55,5):
                x1=int(label[i])
                y1=int(label[i+1])
                x2=x1+int(label[i+2])
                y2=y1+int(label[i+3])
                dim=(x1,y1,x2,y2)

                
                image_croped=image.crop(dim)
                if self.transform:
                    image_croped=self.transform(image_croped)
                images.append(image_croped)
                
            stacked_images=torch.stack(images)
            
            return stacked_images,T_label

batch_size=8
class B3network(nn.Module):
    def __init__(self):
        super(B3network,self).__init__()
        self.classifier=nn.Sequential(nn.Linear(2048,1024),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(),
                                      nn.Linear(1024,512),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(512,8))

    def forward(self,x):
        x=x.view(-1,11,2048)
        x,_=torch.max(x,1)
        x=self.classifier(x)
        return x

model=torchvision.models.resnet50(weights="DEFAULT")

layers=list(model.children())[:-1]
layers.extend([nn.Flatten(),
                nn.Dropout(),
               nn.Linear(2048,1024),
               nn.ReLU(inplace=True),
               nn.Linear(1024,9)])
model=nn.Sequential(*layers)

weights=torch.load("/kaggle/input/datasets/ziadabdallah047/b3-s1-models/B3bestmodel.pth")
newweights=OrderedDict()
for k,v in weights.items():
    name=k[7:] if k.startswith("module.") else k
    newweights[name]=v

model.load_state_dict(newweights)
feature=list(model.children())[:-5]
model1=nn.Sequential(*feature)
for param in model1.parameters():
    param.requires_grad=False
if torch.cuda.device_count()>1:
    model1=nn.DataParallel(model1)

model1.to(device)
model2=B3network()
s2_weights=torch.load("/kaggle/input/notebooks/ziadabdallah047/baseline3/B3_s2_bestmodel.pth")
model2.load_state_dict(s2_weights)
if torch.cuda.device_count()>1:
    model2=nn.DataParallel(model2)
model2.to(device)

test_videos = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]
testset=B3Dataset(test_videos)
testloader=data.DataLoader(testset,batch_size=8)
correct=0
total=0

model1.eval()
model2.eval()
with torch.no_grad():
    for inputs,labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        B,i,c,h,w=inputs.size()
        inputs=inputs.view(B*i,c,h,w)
        feature_extracred=model1(inputs)
        outputs=model2(feature_extracred)
        _,predicted=torch.max(outputs,1)
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)
    print(f"acurracy :{correct/total}")

