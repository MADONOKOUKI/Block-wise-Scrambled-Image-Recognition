import torch
import random
import numpy as np
_rotate = []
_negaposi = []
_reverse = []
_channel = []
random.seed(30)
_shf = []

def rotate(val,p):
  out = val
  if p == 0:
    for i in range(3):
      out[:,:,:,i] = np.rot90(val[:,:,:,i],k=1,axes=(1,2))
  elif p == 1:
    for i in range(3):
      out[:,:,:,i] = np.rot90(val[:,:,:,i],k=2,axes=(1,2))
  elif p == 2:
    for i in range(3):
      out[:,:,:,i] = np.rot90(val[:,:,:,i],k=3,axes=(1,2))
  return out

def negaposi(val,o):
  out = val
  p = np.full((val.shape[0],4,4,3),255)
  if o == 0:
    for i in range(3):
      out[:,:,:,i] = p[:,:,:,i] - val[:,:,:,i]
  return out

def reverse(val,p):
  out = val
  if p == 0:
    for i in range(3):
      out[:,:,:,i] = np.flip(val[:,:,:,i],axis=1)
  elif p == 1:
    for i in range(3):
      out[:,:,:,i] = np.flip(val[:,:,:,i],axis=2)
  return out

def channel_change(val,p):
  out = val
  if p == 1:
    out[:,:,:,0] = val[:,:,:,1]
    out[:,:,:,1] = val[:,:,:,0]
  elif p == 2:
    out[:,:,:,0] = val[:,:,:,2]
    out[:,:,:,2] = val[:,:,:,0]
  elif p == 3:
    out[:,:,:,1] = val[:,:,:,2]
    out[:,:,:,2] = val[:,:,:,1]
  elif p == 4:
    out[:,:,:,0] = val[:,:,:,2]
    out[:,:,:,1] = val[:,:,:,0]
    out[:,:,:,2] = val[:,:,:,1]
  elif p == 5:
    out[:,:,:,0] = val[:,:,:,1]
    out[:,:,:,1] = val[:,:,:,2]
    out[:,:,:,2] = val[:,:,:,0]
  return out
for i in range(64):
  x = random.randint(0,3)
  z = random.randint(0,2)
  a = random.randint(0,5)
  if i%2==0:
      _negaposi.append(1)
  else:
      _negaposi.append(0)
  _rotate.append(x)
  _reverse.append(z)
  _channel.append(a)
  _shf.append(i)
random.shuffle(_shf)
random.shuffle(_negaposi)

def EtC_encryption(img):
    img = np.transpose(img * 255.0,(0 ,2 ,3 ,1 ))
#        img = np.transpose(img,(0, 2, 3, 1))

    for i in range(8):
      for j in range(8):
        # rotate
        img[:,i*4:(i+1)*4,j*4:(j+1)*4,:] = rotate(img[:,i*4:(i+1)*4,j*4:(j+1)*4,:],_rotate[i*8+j])
        # negaposi
        img[:,i*4:(i+1)*4,j*4:(j+1)*4,:] = negaposi(img[:,i*4:(i+1)*4,j*4:(j+1)*4,:],_negaposi[i*8+j])
        # reverse
        img[:,i*4:(i+1)*4,j*4:(j+1)*4,:] = reverse(img[:,i*4:(i+1)*4,j*4:(j+1)*4,:],_reverse[i*8+j])
        # channel change
        img[:,i*4:(i+1)*4,j*4:(j+1)*4,:] = channel_change(img[:,i*4:(i+1)*4,j*4:(j+1)*4,:],_channel[i*8+j])
    tmp = img.copy()

    for i in range(64):
      l = _shf[i]//8
      r = _shf[i]%8
      a = i//8
      b = i%8
      img[:,a*4:(a+1)*4,b*4:(b+1)*4,:] = tmp[:,l*4:(l+1)*4,r*4:(r+1)*4,:].copy()
    img = torch.from_numpy(np.transpose(img,(0 , 3, 1, 2))/255.0)
    return img
