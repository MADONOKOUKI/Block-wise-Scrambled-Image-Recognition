def block_location_shuffle(permutation,img):
    tmp = img.copy()
    for i in range(64):
      l = permutation[i]//8
      r = permutation[i]%8
      a = i//8
      b = i%8
      img[:,:,a*4:(a+1)*4,b*4:(b+1)*4] , tmp[:,:,l*4:(l+1)*4,r*4:(r+1)*4] = tmp[:,:,l*4:(l+1)*4,r*4:(r+1)*4] , img[:,:,a*4:(a+1)*4,b*4:(b+1)*4]
    return img
