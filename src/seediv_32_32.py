import torch
import torch.nn as nn
import numpy as np
import math

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

input_dim = 310
batch_train = 32
batch_test = 32

d_model=512
d_q=d_k=d_v=64
n_heads=8
d_ff=1024
n_layers = 6
epochs = 200
learning_rate = 1e-5


def loadTrainData(path):
    with open(path+'data_training.npy', 'rb') as fileTrain:
        x_train  = np.load(fileTrain)

    with open(path+'label_training.npy', 'rb') as fileTrainL:
        y_train  = np.load(fileTrainL)

    
    x_train = x_train.astype(np.float32)
    x_train = torch.from_numpy(x_train)
    y_train = y_train.astype(np.float32)
    y_train = torch.from_numpy(y_train)

    sample_num_train = x_train.shape[0]
    batch_number_train = sample_num_train // batch_train

    dataset_train = TensorDataset(x_train, y_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_train, shuffle=False, drop_last=True)
    return batch_number_train,dataloader_train


def loadTestData(path):
    with open(path+'data_testing.npy', 'rb') as fileTest:
        x_test  = np.load(fileTest)

    with open(path+'label_testing.npy', 'rb') as fileTestL:
        y_test  = np.load(fileTestL)


    x_test = x_test.astype(np.float32)
    x_test = torch.from_numpy(x_test)
    y_test = y_test.astype(np.float32)
    y_test = torch.from_numpy(y_test)

    sample_num_test = x_test.shape[0]
    batch_number_test = sample_num_test // batch_test

    dataset_test = TensorDataset(x_test, y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_test, shuffle=False, drop_last=True)
    return batch_number_test,dataloader_test

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2*i / d_model) for i in range(d_model)] for pos in range(max_len)
        ])
        pos_table[1:,0::2] = np.sin(pos_table[1:,0::2])
        pos_table[1:,1::2] = np.cos(pos_table[1:,1::2])
        self.pos_table = torch.FloatTensor(pos_table)
        
    def forward(self,inputs):
        inputs += self.pos_table[:inputs.size(1),:].to(device)
        return self.dropout(inputs)
    
def get_pad_mask(q,k):
    mask = torch.ones(q.shape[0],q.shape[1],k.shape[1])
    mask = mask.to(device)
    return mask

class SelfAttention(nn.Module):
    def __init__(self,d_model,d_q,d_k,d_v,n_heads):
        super().__init__()
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        self.quary = nn.Linear(d_model,d_q*n_heads).to(device)
        self.key = nn.Linear(d_model,d_k*n_heads).to(device)
        self.value = nn.Linear(d_model,d_v*n_heads).to(device)
            
        self.linear = nn.Linear(d_v*n_heads,d_model).to(device)
        self.layernorm = nn.LayerNorm(d_model).to(device)
        #self.dropout = nn.Dropout(0.1)
    def forward(self,q,k,v,mask):
        Q = self.quary(q).to(device)
        K = self.key(k).to(device)
        V = self.value(v).to(device)
        
        Q = Q.view(q.shape[0],q.shape[1],self.n_heads,self.d_q).transpose(1,2)
        K = K.view(k.shape[0],k.shape[1],self.n_heads,self.d_k).transpose(1,2)
        V = V.view(v.shape[0],v.shape[1],self.n_heads,self.d_v).transpose(1,2)
         
        mask = mask.tile(1,self.n_heads,1,1)
        #print("mask.shape:",mask.shape)
        scores = torch.matmul(Q,K.transpose(-1,-2)) / math.sqrt(self.d_k)
        scores = scores.to(device)
        #print("score.shape:",scores.shape)
        scores.masked_fill(mask==0,-1e9)
        attn = nn.Softmax(dim=-1)(scores)
        
        output = torch.matmul(attn,V).to(device)
        output = output.view(output.shape[0],output.shape[2],self.n_heads*self.d_v).to(device)
        q = q.to(device)
        output = self.layernorm(q+self.linear(output)).to(device)
        return output,scores
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__ (self,d_model,d_ff):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d_model,d_ff).to(device)
                               ,nn.ReLU(inplace=True)
                               ,nn.Linear(d_ff,d_model).to(device))
        self.layernorm = nn.LayerNorm(d_model).to(device)
    def forward(self,input):
        output = self.layernorm(self.fc(input)+input)
        return output
    
class Encoder(nn.Module):
    def __init__(self,input_dim,d_model,d_q,d_k,d_v,n_heads,d_ff):
        super().__init__()
        self.embedding = nn.Linear(input_dim,d_model).to(device)
        self.pe = PositionalEncoding(d_model).to(device)
        self.attention = SelfAttention(d_model,d_q,d_k,d_v,n_heads).to(device)
        self.ff = PoswiseFeedForwardNet(d_model,d_ff).to(device)
        
    def forward(self,input,n_layers):
        tmp = input
        input = self.embedding(input).to(device)
        input = self.pe(input).to(device)
        mask = get_pad_mask(tmp,tmp)
        for _ in range(n_layers):
            input,attn = self.attention(input,input,input,mask)   
            input = self.ff(input).to(device)
        return input,attn
    
class Transformer(nn.Module):
    def __init__(self,input_dim,d_model,d_q,d_k,d_v,n_heads,d_ff):
        super().__init__()
        self.encoder_eeg = Encoder(input_dim,d_model,d_q,d_k,d_v,n_heads,d_ff)
        self.pro = nn.Linear(d_model,1).to(device)
    def forward(self,input):
        output, attn = self.encoder_eeg(input,6)
        output = self.pro(output)
        output = output.view(-1,output.size(-1))
        #print("output",output)
        return output,attn
    
    
model = Transformer(input_dim,d_model,d_q,d_k,d_v,n_heads,d_ff).to(device)
class_loss_func = nn.MSELoss()
optimizer_classifier = torch.optim.SGD(model.parameters(), lr=learning_rate)

batch_number_train,dataloader_train = loadTrainData('./SEEDIV/')
batch_number_test,dataloader_test = loadTestData('./SEEDIV/')
accuracy_train_list = []
accuracy_test_list = []
for epoch in range(epochs):
    # 训练
    model.train()

    loss_test = 0
    accuracy_test = 0
    loss_train = 0
    accuracy_train = 0

    best_loss_test = 999  
    best_accuracy_test = 0
    best_loss_train = 999  
    best_accuracy_train = 0

    epoch_loss_train = 0
    epoch_accuracy_train = 0
    for fusion_used,train_label_used in dataloader_train:
        fusion_used= fusion_used.unsqueeze(0).to(device)    
        train_label_used= train_label_used.unsqueeze(0).to(device)
        #print(fusion_used.shape) #(1,32,310)

        output, attn = model(fusion_used)
        output = abs(output)
        
        loss_train = class_loss_func(output, train_label_used.view(-1,1)).to(device) 
        optimizer_classifier.zero_grad()
        loss_train.backward()
        optimizer_classifier.step()
        #loss
        epoch_loss_train += loss_train
        if loss_train < best_loss_train:
            best_loss_train = loss_train
        #accuracy
        accuracy_train = np.sum(np.round(output.reshape(output.shape[0]).detach().cpu().numpy()) == train_label_used.squeeze(0).detach().cpu().numpy()) / output.shape[0]
        epoch_accuracy_train += accuracy_train
        

    #测试
    model.eval()

    epoch_loss_test = 0
    epoch_accuracy_test = 0
    for x,y in dataloader_test:
        x = x.unsqueeze(0).to(device)  
        y = y.unsqueeze(0).to(device)
        output, attn,  = model(x)
        output = abs(output)
        #loss(best loss & average loss)
        loss_test = class_loss_func(output, y.view(-1,1)).to(device) 
        if loss_test < best_loss_test:
            best_loss_test = loss_test
        epoch_loss_test +=  loss_test.item()
        #accuracy
        accuracy_test = np.sum(np.round(output.reshape(output.shape[0]).detach().cpu().numpy()) == y.squeeze(0).detach().cpu().numpy()) / output.shape[0]
        epoch_accuracy_test += accuracy_test
        

    
    print('Epoch: {} -- Train loss: {:.6f} --Train accuracy: {:.6f} -- Test loss: {:.6f} -- Test accuracy: {:.6f}'
          .format(epoch,epoch_loss_train/batch_number_train,epoch_accuracy_train/batch_number_train
                  ,epoch_loss_test/batch_number_test, epoch_accuracy_test/batch_number_test))
    
    #average accuracy mean/std
    accuracy_train_list.append(epoch_accuracy_train/batch_number_train)
    accuracy_test_list.append(epoch_accuracy_test/batch_number_test)

    print('Epoch: {} --Train accuracy mean:{:.6f} --Train accuracy std:{:.6f} --Test accuracy mean:{:.6f} --Test accuracy std:{:.6f}'
            .format(epoch, np.mean(accuracy_train_list), np.var(accuracy_train_list), np.mean(accuracy_test_list), np.var(accuracy_test_list)))

    #model save
    if((epoch+1)%50==0):
        if best_accuracy_test < (epoch_accuracy_test/batch_number_test):
            best_accuracy_test = (epoch_accuracy_test/batch_number_test)
            torch.save({ 'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer_classifier.state_dict(),
                        'loss_train':loss_train,
                        'accuracy_train_list':accuracy_train_list,
                        'accuracy_test_list':accuracy_test_list
                        },'./SEEDIV/model2_32_32.pt')
            print("------------------------------model saved-------------------------------------")
        
    
    

