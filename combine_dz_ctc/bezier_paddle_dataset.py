"""
paddle_dataset
"""
import numpy as np
import random
import struct
import pdb

cfg_feat_dim = 8


def read_lst(lst):
    lst = open(lst)
    while True:
        blk_file = lst.readline()
        if not blk_file:
            break
        blk_file = blk_file.strip()
        desc_file = lst.readline().strip()
        yield blk_file, desc_file


def gen_element(blk_file, desc_file, dtype="int16"):
    blk_buffer = open(blk_file, "rb").read()
    descs = open(desc_file).readlines()[1:]

    bch = {"int16": "h", "int32": "i", "float32": "f", "byte": None}[dtype]
    for desc in descs:
        gstr, _, sbyte, lbyte, nele, dim = desc.strip().split()
        sbyte = int(sbyte)
        lbyte = int(lbyte)
        nele = int(nele)
        dim = int(dim)

        buffer = blk_buffer[sbyte:sbyte + lbyte]
        if bch is None:
            yield buffer
        else:
            # if dtype == "float32" and dim != 8:
            #     pdb.set_trace()
            yield np.array(struct.unpack(bch * nele * dim, buffer)).reshape((nele, dim))


class Dataset():
    """
    dataset class
    """

    def __init__(self, feat_lst, label_lst, label_map_file, batch_size):
        """
        init
        """
        self.feat_lst = feat_lst
        self.label_lst = label_lst
        self.batch_size = batch_size
        self.hz2id_map = dict()
        self.id2hz_map = dict()

        for idx, line in enumerate(open(label_map_file)):
            self.hz2id_map[line.strip()] = idx
        for idx, line in enumerate(open(label_map_file)):
            self.id2hz_map[line.strip()] = idx

        self.data_gen = self.gen_func()

    def reset(self, ):
        """
        reset
        """
        self.data_gen = self.gen_func()

    def gen_func(self, ):
        for feat_meta, label_meta in zip(read_lst(self.feat_lst), read_lst(self.label_lst)):
            feat_gen = gen_element(feat_meta[0], feat_meta[1], "float32")
            label_gen = gen_element(label_meta[0], label_meta[1], "int32")
            for feat, label in zip(feat_gen, label_gen):
                yield feat, label

    def batch_num(self):
        """
        batch num
        """
        return None

    def next(self):
        """
        next
        """
        # print(self.line_idx)

        feats = []
        lod_feat = []
        labels = []
        lod_label = []

        for feat, label in self.data_gen:
            feats.extend(feat)
            labels.extend(label)
            lod_feat.append(len(feat))
            lod_label.append(len(label))
            if len(lod_label) == self.batch_size:
                break

        labels = np.array(labels, dtype=np.int32).reshape([-1, 1])
        lod_label = np.array(lod_label, dtype=np.int32)
        feats = np.array(feats, dtype=np.float32)
        lod_feat = np.array(lod_feat, dtype=np.int32)

        if len(feats.shape) != 2:
            return None, None, None, None
        # feats = feats.reshape([len(feats) // cfg_feat_dim, cfg_feat_dim])

        return labels, lod_label, feats, lod_feat


if __name__ == "__main__":
    dataset = Dataset("feat.lst", "label.lst", "hz_utf.txt", 32)
    dataset.next()

    a, b, c, d = dataset.next()
