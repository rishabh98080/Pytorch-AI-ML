import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n_samples = 1000

X,Y = make_circles(n_samples=n_samples, noise=0.05,random_state=42)

print(X.shape)
print(Y.shape)


import pandas as pd

circles = pd.DataFrame(X,columns=['x1','x2'])
circles['label'] = Y

print(circles.head(10))


# plt.scatter(circles['x1'],circles['x2'],c=circles['label'],cmap=plt.cm.RdYlBu)
# plt.show()



###########################################################################
import torch


X = torch.from_numpy(X).type(torch.float32)
Y = torch.from_numpy(Y).type(torch.float32)


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state = 42)



##################################
###### Model #######

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

from torch import nn

# class CircleModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 =  nn.Linear(in_features=2,out_features=5)
#         self.layer_2 =  nn.Linear(in_features=5,out_features=1)
#     def forward(self,x):
#         return self.layer_2(self.layer_1(x))
torch.manual_seed(42)

#model_0  = CircleModel()
#print(model_0.state_dict())

model_0 = nn.Sequential(
    nn.Linear(in_features=2,out_features=5),
    nn.Linear(in_features=5,out_features= 1)
)

with torch.inference_mode():
    y_untrained_preds = model_0(X_test)
print(f"Length : {len(y_untrained_preds)},Shape : {y_untrained_preds.shape}")
print(f"Length : {len(X_test)},Shape : {X_test.shape}")
print(f"Prediction : {y_untrained_preds[:10]}, Shape : {y_untrained_preds[:10].shape}")
print(f"Test data : {Y_test[:10]}")


### LOss fn and optimizer fn

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params = model_0.parameters(),lr =0.01)


####Accuracy

def accuracy_fn(y_true,y_preds):
    correct = torch.eq(y_true,y_preds).sum().item()
    acc = (correct/len(y_preds)) * 100
    return acc


# ###Training MOdel
# model_0.eval()
# with torch.inference_mode():
#     y_logits = model_0(X_test)
# # print(y_logits[:10],'\n',torch.round(y_logits[:10]))

# y_preds_prob = torch.round(torch.sigmoid(y_logits))

# # print(y_preds_prob[:10],'\n',torch.round(y_preds_prob[:10]))


# ##Full

# y_predict_label = torch.round(torch.sigmoid(model_0(X_test)))


# print(torch.eq(y_preds_prob.squeeze(),y_predict_label.squeeze()))



####Training loop
torch.manual_seed(42)
epochs = 10000

for epoch in range(epochs):
    ## Training Mode
    model_0.train()
    y_logits = model_0(X_train)
    y_prediction = torch.round(torch.sigmoid(y_logits)).squeeze(dim = 1)
    
    ### Loss calc

    Loss = loss_fn(y_logits.squeeze(),Y_train)

    optimizer.zero_grad()

    Loss.backward()

    optimizer.step()

    print(model_0.state_dict())

model_0.eval()
with torch.inference_mode():
    y_test_predictions = model_0(X_test)

print(y_test_predictions,Y_test)

print(X_test[:,0][:10],X_test[:,1][:10],X_test[:10])
plt.scatter(X_test[:,0],X_test[:,1],c = y_test_predictions,cmap=plt.cm.RdYlBu)
plt.show()