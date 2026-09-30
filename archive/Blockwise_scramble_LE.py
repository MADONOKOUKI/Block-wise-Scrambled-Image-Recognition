from learnable_encryption import BlockScramble
import numpy as np
def blockwise_scramble(imgs,key_size = 4):
  x_stack  = None
  for k in range(8):
    tmp = None
   # x_stack = None
    for j in range(8):
      key_file = 'key4/'+str(0)+'_.pkl'
      bs = BlockScramble( key_file )
      out = np.transpose(imgs,(0, 2, 3, 1))
      out = out[:,k*4:(k+1)*4,j*4:(j+1)*4,:]
      out = bs.Scramble(out.reshape([out.shape[0],4,4,3])).reshape([out.shape[0],4,4,3])
      if tmp is None:
        tmp = out
      else:
        tmp = np.concatenate((tmp,out),axis=2)
    if x_stack is None:
      x_stack = tmp
    else:
      x_stack = np.concatenate((x_stack,tmp),axis=1)
  return x_stack
