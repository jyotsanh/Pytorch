import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset



x_train = [1,2,3,4]
y_train = [1,2,3,4]
x_test = [1,2,3,4]
y_test = [2,3,4,5]

class Data(Dataset):
    def __init__(self,x_train,y_train):
        super().__init__()
        self.x = torch.from_numpy(x_train).type(torch.float32)
        self.y = torch.from_numpy(y_train).type(torch.float32)

class Logistic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=10,out_features=32
        )
        self.linear2 = nn.Linear(
            in_features=32,out_features=16
        )
        
        self.linear3 = nn.Linear(
            in_features=16,out_features=1
        )
        self.relu = nn.ReLU()
        
    def forward(self,x:torch.tensor):
        x  = self.linear1(x)
        x = self.relu(x)
        x  = self.linear2(x)
        x = self.relu(x)
        x  = self.linear3(x)


model = Logistic()
learning_rate = 0.01
loss_fn = nn.BCELoss()
optim = torch.optim.SGD(
    model.parameters(),lr=learning_rate
)

loader = Data(x_train,y_train)

train_loader = DataLoader(loader,batch_size=10,shuffle=True)


epochs = 10
for epoch in range(epochs):
    for batch_X,batch_Y in train_loader:
        model.train()
        y_pred = model(batch_X)
        loss = loss_fn(y_pred,batch_Y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    model.eval()
        
    with torch.inference_mode():
        y_pred = model(x_test)
        loss = loss_fn(y_pred,y_test)
        
        print(f"Epoch : {epoch+1} , BCE Error:{loss} ")
            