import torch
import torch.nn as nn
import numpy as np
import math

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

input_dim = 310
input_bits = 4
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

class AlphaInit(nn.Parameter):
    def __init__(self,tensor):
        super().__new__(nn.Parameter,data=tensor)
        self.initialized = False
    def _initialize(self, init_tensor):
        self.data.copy_(init_tensor)
        self.initialized = True
    def initialize_wrapper(self, tensor, num_bits, symmetric):
        Qp = 2 ** (num_bits-1) - 1 if symmetric else 2 ** (num_bits) -  1
        if Qp == 0:
            Qp = 1.0
        init_val = 2 * torch.abs(tensor).mean() / math.sqrt(Qp) if symmetric else 4 * torch.abs(tensor).mean() / math.sqrt(Qp)
        self._initialize(init_val)

class TwnQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, layerwise):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else:  # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None
    
class BwnQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,layerwise):
        ctx.save_for_backward(input)
        if layerwise:
            s = input.size()
            m = input.norm(p=1).div(input.nelement())
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))
        else:
            n = input[0].nelement()
            s = input.size()
            m = input.norm(1,1,keepdim=True).div(n)
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))
        return result
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None
    
class ElasticQuantBinarizerSigned(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,alpha,num_bits=1):
        ctx.num_bits = num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits -1) #-2,-4,-8
            Qp = 2 ** (num_bits - 1) - 1 #1,3,7
            
        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input,num_bits,symmetric=True)  
        alpha = torch.where(alpha > eps, alpha, eps)
        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input,alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = input.sign().to(device)
        else:
            q_w = (input/alpha).round().clamp(Qn,Qp).to(device) 
        w_q = (q_w * alpha).to(device)     
        return w_q
    @staticmethod
    def backward(ctx, grad_output):
        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.num_bits == 1:
            grad_alpha = ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        else:
            grad_alpha = ((indicate_small*Qn+indicate_big*Qp+indicate_middle*(-q_w+q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None
 
class ElasticQuantBinarizerUnsigned(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,alpha,num_bits=1):
        ctx.num_bits = num_bits
        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val
        
        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input,num_bits,symmetric=False)  
        alpha = torch.where(alpha > eps, alpha, eps)
        
        grad_scale = 1.0 / math.sqrt(input.numel()*Qp)
        #print("grad_scale",grad_scale)
        ctx.save_for_backward(input_,alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_/alpha).round().clamp(Qn,Qp).to(device)
        w_q = (q_w * alpha).to(device)
        if num_bits != 1:
            w_q = w_q + min_val
        return w_q
    @staticmethod
    def backward(ctx, grad_output):
        input_,alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_/alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        grad_alpha = ((indicate_small*Qn+indicate_big*Qp+indicate_middle*(-q_w+q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None
    
class QuantizeLinear(nn.Linear):
    def __init__(self,*kargs,weight_layerwise=True,input_bits=8,**kwargs):
        super().__init__(*kargs,**kwargs)
        self.weight_layerwise = weight_layerwise
        self.input_bits = input_bits
        self.input_clip_val = AlphaInit(torch.tensor(1.0))
    def forward(self,input):
        quant_fn = BwnQuantizer
        #quant_fn = TwnQuantizer
        weight = quant_fn.apply(self.weight,self.weight_layerwise)
        act_quant_fn = ElasticQuantBinarizerSigned
        input = act_quant_fn.apply(input,self.input_clip_val,self.input_bits)
        out = nn.functional.linear(input,weight)
        return out

class LearnableBias(nn.Module):
    def __init__(self,out_chn):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn),requires_grad=True).to(device)  
    def forward(self,x):
        x = x.to(device)
        out = (x + self.bias.expand_as(x)).to(device)
        return out
    
class SelfAttention(nn.Module):
    def __init__(self,d_model,d_q,d_k,d_v,n_heads,input_bits):
        super().__init__()
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.input_bits = input_bits
        
        self.quary = QuantizeLinear(d_model,d_q*n_heads,input_bits=self.input_bits).to(device)
        self.key = QuantizeLinear(d_model,d_k*n_heads,input_bits=self.input_bits).to(device)
        self.value = QuantizeLinear(d_model,d_v*n_heads,input_bits=self.input_bits).to(device)
        
        self.move_q = LearnableBias(d_q*n_heads)
        self.move_k = LearnableBias(d_k*n_heads)
        self.move_v = LearnableBias(d_v*n_heads)
        
        if input_bits < 32:
            self.clip_query = AlphaInit(torch.tensor(1.0))
            self.clip_key = AlphaInit(torch.tensor(1.0))
            self.clip_value = AlphaInit(torch.tensor(1.0))
            self.clip_attn = AlphaInit(torch.tensor(1.0))
            
        self.linear = QuantizeLinear(d_v*n_heads,d_model).to(device)
        self.layernorm = nn.LayerNorm(d_model).to(device)
        #self.dropout = nn.Dropout(0.1)
    def forward(self,q,k,v,mask):
        Q = self.quary(q).to(device)
        K = self.key(k).to(device)
        V = self.value(v).to(device)
        
        if self.input_bits < 32:
            Q = self.move_q(Q)
            K = self.move_k(K)
            V = self.move_v(V)
            
        Q = Q.view(q.shape[0],q.shape[1],self.n_heads,self.d_q).transpose(1,2)
        #print("Q.shape",Q.shape)
        K = K.view(k.shape[0],k.shape[1],self.n_heads,self.d_k).transpose(1,2)
        V = V.view(v.shape[0],v.shape[1],self.n_heads,self.d_v).transpose(1,2)
        
        act_quant_fn = ElasticQuantBinarizerSigned
        if self.input_bits < 32:
            Q = act_quant_fn.apply(Q,self.clip_query,self.input_bits)
            K = act_quant_fn.apply(K,self.clip_key,self.input_bits)
            V = act_quant_fn.apply(V,self.clip_value,self.input_bits)
       
        mask = mask.tile(1,self.n_heads,1,1)
        #print("mask.shape",mask.shape)
        scores = torch.matmul(Q,K.transpose(-1,-2)) / math.sqrt(self.d_k)
        #print("scores.shape",scores.shape)
        scores = scores.to(device)
        scores.masked_fill(mask==0,-1e9)
        attn = nn.Softmax(dim=-1)(scores)
        
        act_quant_fn_attn = ElasticQuantBinarizerUnsigned
        if self.input_bits < 32:
            attn = act_quant_fn_attn.apply(attn,self.clip_attn,self.input_bits)
        output = torch.matmul(attn,V).to(device)
        output = output.view(output.shape[0],output.shape[2],self.n_heads*self.d_v).to(device)
        q = q.to(device)
        output = self.layernorm(q+self.linear(output)).to(device)
        return output,scores
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.fc = nn.Sequential(QuantizeLinear(d_model,d_ff).to(device)
                               ,nn.ReLU(inplace=True)
                               ,QuantizeLinear(d_ff,d_model).to(device))
        self.layernorm = nn.LayerNorm(d_model).to(device)
    def forward(self,input):
        output = self.layernorm(self.fc(input)+input)
        return output
    
class Encoder(nn.Module):
    def __init__(self,input_dim,d_model,d_q,d_k,d_v,n_heads,input_bits,d_ff):
        super().__init__()
        self.embedding = nn.Linear(input_dim,d_model).to(device)
        self.pe = PositionalEncoding(d_model).to(device)
        self.attention = SelfAttention(d_model,d_q,d_k,d_v,n_heads,input_bits).to(device)
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
    def __init__(self,input_dim,d_model,d_q,d_k,d_v,n_heads,input_bits,d_ff):
        super().__init__()
        self.encoder = Encoder(input_dim,d_model,d_q,d_k,d_v,n_heads,input_bits,d_ff)
        self.pro = nn.Linear(d_model,1).to(device)
    def forward(self,input):
        output, attn = self.encoder(input,6)
        logits = self.pro(output)
        logits = logits.view(-1,logits.size(-1))
        return logits,attn
    
model = Transformer(input_dim,d_model,d_q,d_k,d_v,n_heads,input_bits,d_ff).to(device)
class_loss_func = nn.MSELoss()
optimizer_classifier = torch.optim.SGD(model.parameters(), lr=learning_rate)

batch_number_train,dataloader_train = loadTrainData('./SEED/')
batch_number_test,dataloader_test = loadTestData('./SEED/')
accuracy_train_list = []
accuracy_test_list = []
# checkpoint = torch.load('./SEED/model_bwn_1_4.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer_classifier.load_state_dict(checkpoint['optimizer_state_dict'])
# last_epoch = checkpoint['epoch']
# loss_train = checkpoint['loss_train']
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
                        },'./SEED/model_bwn_1_4_.pt')
            print("------------------------------model saved-------------------------------------")
        
    