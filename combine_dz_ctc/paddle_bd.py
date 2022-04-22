"""
paddle_dataset
"""

import numpy as np
import random
import struct
import datetime
import sys

cfg_feat_dim = 8


class Dataset():
    """
    dataset class
    """
    
    def __init__(self):
        """
        init
        """
        
        self.idx = 0
        self.batch_size = 8
        self.data = []
        
        
    def read_weight_data(self, path):
        """
        read weight data
        """
        
        weis = []
        datas = []
        
        if sys.version_info < (3, 0):
            with open(path, 'r') as in_f:
                lines = in_f.readlines()
                for line in lines:
                    line = line.strip()
                    line = line.decode('utf-8')
                    datas.append(line)
                         
        else:
            with open(path, 'r', encoding='utf-8') as in_f:
                lines = in_f.readlines()
                for line in lines:
                    line = line.strip()
                    datas.append(line)

        for i in range(0, len(datas), 3):
            wname = datas[i]
            wshape = [int(x) for x in datas[i + 1].split(' ')]
            wdata = [float(x) for x in datas[i + 2].split(' ')]
            
            print(wname, wshape)
            
            weis.append(np.array(wdata, dtype=np.float32).reshape(wshape))
                         
        return weis
        

    def open(self, label_path, feat_path, batch_size):
        """
        open
        """
        
        t1 = datetime.datetime.now()
        
        self.batch_size = batch_size
        self.idx = 0
        self.datas = []
         
        with open(feat_path, 'rb') as in_f:
            
            while True:
                plen = in_f.read(4)
                if plen is None or len(plen) != 4:
                    break
                
                alllen = struct.unpack('i', plen)[0]
                label_len = alllen & 0xFFFF
                seq_len = (alllen >> 16) & 0xFFFF                
               
                label_raw = in_f.read(4 * label_len)
                seq_raw = in_f.read(4 * seq_len * cfg_feat_dim)
                
                if (len(seq_raw) != 4 * seq_len * cfg_feat_dim):
                    break
                self.datas.append((label_raw, seq_raw))
                
        t2 = datetime.datetime.now()
        
        print(t2 - t1)
    
    
    def reset(self, shuffle):
        """
        reset
        """
        
        self.idx = 0
        if (shuffle):
            random.shuffle(self.datas)
            
            
    def batch_num(self):
        """
        batch num
        """
        
        return len(self.datas) // self.batch_size
            
    
    def next(self):
        """
        next
        """
        
        sb_code = []
        for _ in range(4):
            sb_code.append(None)
        
        if self.idx + self.batch_size >= len(self.datas):
            return sb_code
        
        lod_label = []
        lod_feat = []
        labels = []
        feats = []
        
        bid = self.idx
        eid = bid + self.batch_size
        
        while bid < eid:
            
            label, feat = self.datas[bid]
            label = np.frombuffer(label, dtype=np.int32).reshape([len(label) // 4, 1])
            feat = np.frombuffer(feat, dtype=np.float32).reshape([len(feat) // 4 // cfg_feat_dim, cfg_feat_dim])
            lod_label.append(label.shape[0])
            lod_feat.append(feat.shape[0])
            labels.append(label)
            feats.append(feat)
            
            bid += 1
        
        self.idx += self.batch_size
        
        labels = np.vstack(labels)
        lod_label = np.array(lod_label, dtype = np.int32)
        feats = np.vstack(feats)
        lod_feat = np.array(lod_feat, dtype = np.int32)
        
        sb_code = []
        sb_code.append(labels)
        sb_code.append(lod_label)
        sb_code.append(feats)
        sb_code.append(lod_feat)
        
        return sb_code