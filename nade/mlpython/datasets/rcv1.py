# Copyright 2011 Hugo Larochelle. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
# 
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle.

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False,dtype=np.float64):
    """
    Loads the RCV1 dataset. This is actually a smaller version of it, with 150 inputs
    and binary targets.

    The data is given by a dictionary mapping from strings
    'train', 'valid' and 'test' to the associated pair of data and metadata.
    
    Defined metadata: 
    - 'input_size'
    - 'targets'
    - 'length'

    References: Tractable Multivariate Binary Density Estimation and the Restricted Boltzmann Forest
                Larochelle, Bengio and Turian
                link: http://www.cs.toronto.edu/~larocheh/publications/NECO-10-09-1100R2-PDF.pdf

                LIBSVM Data: Classification, Regression, and Multi-label (web page)
                link: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
    """
    
    input_size=150
    dir_path = os.path.expanduser(dir_path)
    targets = set([0,1])
    target_mapping = {'0':0,'1':1}
    def convert_target(target):
        return target_mapping[target]

    def load_line(line):
        tokens = line.split()
        return (np.array([int(i) for i in tokens[:-1]]),int(tokens[-1]))

    train_file,valid_file,test_file = [os.path.join(dir_path, 'rcv1_all_subset.binary_' + ds + '_voc_150.amat') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [40000,10000,150000]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[dtype,int],l) for d,l in zip([train,valid,test],lengths)]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                              'length':l,'targets':targets} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):

    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/rcv1/rcv1_all_subset.binary_train_voc_150.amat',os.path.join(dir_path,'rcv1_all_subset.binary_train_voc_150.amat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/rcv1/rcv1_all_subset.binary_valid_voc_150.amat',os.path.join(dir_path,'rcv1_all_subset.binary_valid_voc_150.amat'))
    urllib.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/rcv1/rcv1_all_subset.binary_test_voc_150.amat',os.path.join(dir_path,'rcv1_all_subset.binary_test_voc_150.amat'))
    print 'Done                     '
