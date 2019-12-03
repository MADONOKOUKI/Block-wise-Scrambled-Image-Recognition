#!/usr/bin/env python
# -*- coding: utf-8 -*-
#import tensorflow as tf
import numpy as np
import math
try:
    import cPickle as pickle
except:
    import pickle
import os
class BlockScramble:
    def __init__( self, blockSize_filename ):
        if( isinstance( blockSize_filename, str ) ):
            self.load( blockSize_filename )
        else:
            self.blockSize = blockSize_filename
            key = self.genKey()
            self.setKey( key )

    def setKey( self, key ):
        self.key = key
        self.rev = ( key > key.size/2 )
        self.invKey = np.argsort(key)
    
    def load( self, filename ):
        fin = open(filename, 'rb')
        self.blockSize, self.key = pickle.load( fin )
        fin.close()
        
        self.setKey( self.key )
    
    def save( self, filename ): # pkl
        fout = open(filename, 'wb')
        pickle.dump( [self.blockSize, self.key], fout )
        fout.close()        
    
    def genKey( self ):
        key = self.blockSize[0] * self.blockSize[1]*self.blockSize[2]
        key = np.arange(key*2, dtype=np.uint32)
        np.random.shuffle(key)
        return key
        
    def padding( self, X ): # X is [datanum, width, height, channel]
        s = X.shape
        
        t = s[1] / self.blockSize[0]
        d = t - math.floor(t)
        if( d > 0 ):
            paddingSize = self.blockSize[0] * ( math.floor(t) + 1 ) - s[1]
            padding = X[:,-1:,:,:]
            padding = np.tile( padding, (1, paddingSize, 1, 1 ) )
            X = np.concatenate( (X, padding), axis = 1 )

        t = s[2] / self.blockSize[1]
        d = t - math.floor(t)
        if( d > 0 ):
            paddingSize = self.blockSize[1] * ( math.floor(t) + 1 ) - s[2]
            padding = X[:,:,-1:,:]
            padding = np.tile( padding, (1, 1, paddingSize, 1 ) )
            X = np.concatenate( (X, padding), axis = 2 )
        
        return X
    
    def Scramble(self, X):
        XX = (X * 255).astype(np.uint8)
        XX = self.doScramble(XX, self.key, self.rev)
        return XX.astype('float32')/255.0
        
    def Decramble(self, X):
        XX = (X * 255).astype(np.uint8)
        XX = self.doScramble(XX, self.invKey, self.rev)
        return XX.astype('float32')/255.0
    
    def doScramble(self, X, ord, rev): # X should be uint8
        s = X.shape
        #print(s)
       # print(self.blockSize)
        assert( X.dtype == np.uint8 )
        assert( s[1] % self.blockSize[0] == 0 )
        assert( s[2] % self.blockSize[1] == 0 )
        assert( s[3] == self.blockSize[2] )
        numBlock = np.int32( [ s[1] / self.blockSize[0], s[2] / self.blockSize[1] ] );
        numCh = self.blockSize[2];
        
        X = np.reshape( X, ( s[0], numBlock[0], self.blockSize[0], numBlock[1], self.blockSize[1],  numCh ) )
        X = np.transpose( X, (0, 1, 3, 2, 4, 5) )
        X = np.reshape( X, ( s[0], numBlock[0], numBlock[1], self.blockSize[0] * self.blockSize[1] * numCh ) )
        d = self.blockSize[0] * self.blockSize[1] * numCh;
       # print(X)
       # print(0xF)
        X0 = X & 0xF # あまりが入る（/16)
       # print(X0)
        X1 = X >> 4 # 16で割ったときの商がはいる
      #  print(X1)
        X = np.concatenate( (X0,X1), axis=3 )
        
        X[:,:,:,rev] = ( 15 - X[:,:,:,rev].astype(np.int32) ).astype(np.uint8)
     #   print(ord)
        X = X[:,:,:,ord]
        X[:,:,:,rev] = ( 15 - X[:,:,:,rev].astype(np.int32) ).astype(np.uint8)

        X0 = X[:,:,:,:d]
        X1 = X[:,:,:,d:]
        X = ( X1 << 4 ) + X0
        
        X = np.reshape( X, ( s[0], numBlock[0], numBlock[1], self.blockSize[0], self.blockSize[1], numCh ) )
        X = np.transpose( X, ( 0, 1, 3, 2, 4, 5) )
        X = np.reshape( X, ( s[0], numBlock[0] * self.blockSize[0], numBlock[1] * self.blockSize[1], numCh ) );
        
        return X

if( __name__ == '__main__' ):
    from PIL import Image
    import os
    import scipy.misc
    from matplotlib import cm
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=0
 #   config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically
#    sess = tf.Session(config = config)
    im = Image.open('lena.png')
    data = np.asarray(im, dtype=np.uint8)
    data = np.reshape( data, (1,)+data.shape )
    print(data.shape)
    
    key_file = 'key16/keys1.pkl'
    
    if( os.path.exists(key_file) ):
        bs = BlockScramble( key_file )
    else:
        bs = BlockScramble( [16,16,3] )
        bs.save(key_file)
    
    data = bs.padding( data )
    print(data.shape)
    
    im = Image.fromarray( data[0,:,:,:] )
    im.save('test_bs1.png')
    print(data.shape)
    data = bs.Scramble( data )
    print(data.shape)
    #array_resized_image = data[0,:,:,:]
    #scipy.misc.imsave("test_bs2.png", array_resized_image)
    #im = Image.fromarray( data[0,:,:,:] ,mode='F')
    im = Image.fromarray(np.uint8(cm.gist_earth(data[0,:,:,:],bytes=True))*255)
    im.save('test_bs2.png')

    data = bs.Decramble( data )
    print(data.shape)
    #array_resized_image = data[0,:,:,:]
    #scipy.misc.imsave("test_bs3.png", array_resized_image)
    im = Image.fromarray(np.uint8(cm.gist_earth(data[0,:,:,:],bytes=True))*255)
    im.save('test_bs3.png')
    



