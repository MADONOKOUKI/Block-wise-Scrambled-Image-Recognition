# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from shakedrop import ShakeDrop

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention

class ShakeBasicBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, p_shakedrop=1.0):
        super(ShakeBasicBlock, self).__init__()
        self.downsampled = stride == 2
        self.branch = self._make_branch(in_ch, out_ch, stride=stride)
        self.shortcut = not self.downsampled and None or nn.AvgPool2d(2)
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        h = self.branch(x)
        h = self.shake_drop(h)
        h0 = x if not self.downsampled else self.shortcut(x)
        pad_zero = Variable(torch.zeros(h0.size(0), h.size(1) - h0.size(1), h0.size(2), h0.size(3)).float()).cuda()
        h0 = torch.cat([h0, pad_zero], dim=1)

        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
 #   U = nn.init.ones_(torch.nn.Parameter(torch.empty(64, 64, dtype=torch.float),requires_grad=True)).cuda()
#    return -Variable(torch.log(-torch.log(1.0 + eps) + eps))
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
   #y = logits + 1.0 
#    return F.softmax(torch.log(logits) / temperature,dim=0)
    return F.softmax(logits / temperature, dim=0)

def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    return y
#    return y.view( latent_dim , categorical_dim)

latent_dim = 64
categorical_dim = 64 # one-of-K vector

#@weak_module
class OriginalLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=False):
        super(OriginalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        #self.weight = torch.abs(self.weight)
    def reset_parameters(self):
        # sample from uniform distribution
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            nn.init.uniform_(self.bias,-bound, bound)
        #self.weight = torch.abs(self.weight)
 #   @weak_script_method
    def forward(self, input):
      weight = self.weight
      if input.dim() == 2 and self.bias is not None:
        # fused op is marginally faster
        #weight = F.softmax(F.softmax(F.relu(self.weight),dim=0),dim=1)
        ret = torch.addmm(self.bias, input,weight)
      else:
        output = input.matmul(weight)
        if self.bias is not None:
            output += self.bias
        ret = output
      return ret,weight
#      return ret,F.relu(self.weight)
      #  return linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class ShakePyramidNet(nn.Module):

    def __init__(self, depth=110, alpha=270, label=10):
        super(ShakePyramidNet, self).__init__()
        in_ch = 16
        # for BasicBlock
        n_units = (depth - 2) // 6
        in_chs = [in_ch] + [in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]
        block = ShakeBasicBlock

        self.in_chs, self.u_idx = in_chs, 0
        self.ps_shakedrop = [1 - (1.0 - (0.5 / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]
        self.ll = nn.LeakyReLU(0.2) # NN.UTILS.SPECTRAL_NORM
        self.convs0 =  nn.ModuleList([nn.utils.spectral_norm(nn.Conv2d(3, 16*16, kernel_size=4, stride=4, padding=0, bias=False)) for _ in range(64)])
        self.bns0 = nn.ModuleList([nn.BatchNorm2d(16*16) for _ in range(64)])
        self.pixelshuffles = nn.ModuleList([nn.PixelShuffle(4) for _ in range(64)])
        self.attn1 = Self_Attn(256, 'relu')
        self.u = nn.init.constant_(torch.nn.Parameter(torch.empty(4096, dtype=torch.float),requires_grad=True),1.00/64)
        
        self.ml1 = nn.Linear(4096,4096)
        self.ml2 = nn.Linear(4096,4096)

        self.matrix = OriginalLinear(64,64)
        #Multu Layer Perceptron
        self.sm = nn.Softmax()
        self.fc1 = nn.Linear(64,64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64,64)
        self.bn3 = nn.BatchNorm1d(64)

        self.pixsh = nn.PixelShuffle(4)
        self.c_in = nn.Conv2d(16, in_chs[0], 3, padding=1)
        self.bn_in = nn.BatchNorm2d(in_chs[0])
        self.layer1 = self._make_layer(n_units, block, 1)
        self.layer2 = self._make_layer(n_units, block, 2)
        self.layer3 = self._make_layer(n_units, block, 2)
        self.bn_out = nn.BatchNorm2d(in_chs[-1])
        self.fc_out = nn.Linear(in_chs[-1], label)

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x_stack = None
        idx = 0
        for i in range(8):
          tmp = None
          for j in range(8):
            out = self.ll(self.bns0[idx](self.convs0[idx](x[:,:,i*4:(i+1)*4,j*4:(j+1)*4]))).view(-1,256,1)
            if tmp is None:
              tmp = out
            else:
              tmp = torch.cat([tmp,out],dim=2)
            idx = idx + 1
          if x_stack is None:
            x_stack = tmp
          else:
            x_stack = torch.cat([x_stack,tmp],dim=2)
        h = x_stack
        pros = h
        stack = None
        stack_recon = None
        stack,mat = self.matrix(pros)
        stack_recon = torch.matmul(mat,mat.t())
        x_stack = None
        for i in range(8):
          tmp = None
          for j in range(8): 
            out = stack[:,:,i*8+j].contiguous().view(-1,256,1,1)
            if tmp is None:
              tmp = out
            else:
              tmp = torch.cat([tmp,out],dim=3)
          if x_stack is None:
            x_stack = tmp
          else:
            x_stack = torch.cat([x_stack,tmp],dim=2) 
        h = x_stack
        h = self.pixsh(h)
        
        feature = h
        h = self.bn_in(self.c_in(h))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(self.bn_out(h))
        h = F.avg_pool2d(h, 8)
        h = h.view(h.size(0), -1)
        h = self.fc_out(h)
        return h,mat,feature
     #   return h,feature,h,torch.norm((torch.eye(64).cuda()-stack_recon)).sum()/(64*64),feature/(64*64),mat

    def _make_layer(self, n_units, block, stride=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(block(self.in_chs[self.u_idx], self.in_chs[self.u_idx+1],
                                stride, self.ps_shakedrop[self.u_idx]))
            self.u_idx, stride = self.u_idx + 1, 1
        return nn.Sequential(*layers)

