import torch

x = torch.tensor([1,2,3,4,5,6],dtype=torch.float32)
y = torch.tensor([2,4,6,8,10,12],dtype=torch.float32)


w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)


def forward(x):
    return w*x
#forward pass and backward pass

def loss(y,y_hat):
    return ((y_hat - y)**2).mean()


learning_rate = 0.01
n_itters = 20


for epoch in range(n_itters):
    y_pred = forward(x)
    l = loss(y,y_pred)
    
    #gradient pass
    l.backward() # dl/dw
    
    
    with torch.no_grad():
        w -= learning_rate*w.grad
    
        w.grad.zero_()
    
    if epoch % 2 == 0:
        print(f"epoch {epoch+1} : w = {w:.3f} , loss = {l:.8f}")
        
print(f"prediction after training =  {forward(20):.3f}")