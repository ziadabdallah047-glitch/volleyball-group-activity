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
        if self.transform:
            image=self.transform(image)

        return image,label

device=("cuda" if torch.cuda.is_available() else "cpu")
model=torchvision.models.resnet50(weights="DEFAULT")
layers=list(model.children())[:-1]
layers.extend([nn.Flatten(),
                nn.Dropout()
               ,nn.Linear(2048,1024),
               nn.ReLU(inplace=True),
               nn.Linear(1024,8)])
model=nn.Sequential(*layers)
model.to(device)

test_videos = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]
testset=B1DataSet(test_videos,train=False)
testloader=data.DataLoader(testset,batch_size=32)
correct=0
total=0
weights19=torch.load("model19.pth")
model.load_state_dict(weights19)
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
