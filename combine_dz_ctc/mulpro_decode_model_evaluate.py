"""
paddle.1.8.2
"""

import os
import platform
import paddle.fluid as fluid
import bezier_paddle_dataset as ds
import numpy as np
import time
import pdb
import logging
import pickle
from multiprocessing import Process, Queue, Lock
from argparse import ArgumentParser

cfg_use_gpu = True

cfg_lr = 0.0
cfg_dropout_rate = 0.10

cfg_lstm_dim = 128
cfg_lstm_num = 4
cfg_lstm_use_peepholdes = False

cfg_fc_dim = 128
cfg_input_dim = 8
cfg_output_dim = 28800
cfg_ctc_output_dim = 6400

cfg_batch_size = 64
cfg_max_epoch = 1

cfg_ini_model_path = "./model"

tst_input = None
tst_output = None

data_reader = ds.Dataset("feat.lst", "label.lst", "hz_utf.txt", cfg_batch_size)


def beam_decode(y, beam_size=6):
    # y是个二维数组，记录了所有时刻的所有项的概率
    T, V = y.shape
    # 将所有的y中值改为log是为了防止溢出，因为最后得到的p是y1..yn连乘，且yi都在0到1之间，可能会导致下溢出
    # 改成log(y)以后就变成连加了，这样就防止了下溢出
    log_y = np.log(y+1e-9)
    # 初始的beam
    beam = [([], 0)]
    # 遍历所有时刻t
    for t in range(T):
        # 每个时刻先初始化一个new_beam
        new_beam = []
        # 遍历beam
        for prefix, score in beam:
            # 对于一个时刻中的每一项(一共V项)
            for i in range(V):
                # 记录添加的新项是这个时刻的第几项，对应的概率(log形式的)加上新的这项log形式的概率(本来是乘的，改成log就是加)
                new_prefix = prefix + [i]
                new_score = score + log_y[t, i]
                # new_beam记录了对于beam中某一项，将这个项分别加上新的时刻中的每一项后的概率
                new_beam.append((new_prefix, new_score))
        # 给new_beam按score排序
        new_beam.sort(key=lambda x: x[1], reverse=True)
        # beam即为new_beam中概率最大的beam_size个路径
        beam = new_beam[:beam_size]
        # pdb.set_trace()
    beam = [( x[0], np.exp(x[1]) ) for x in beam]
    return beam



def batch_generator_creator():
    """
    read batch data
    """
    
    def __reader__():
        label, lod_label, feat, lod_feat = data_reader.next()
        while label is not None:
            
            iinputs = fluid.create_lod_tensor(feat, [lod_feat], train_place)
            ilabels = fluid.create_lod_tensor(label, [lod_label], train_place)
            
            yield iinputs, ilabels
            
            label, lod_label, feat, lod_feat = data_reader.next()

    return __reader__



def lstm_cell(inputs, dim, use_ph, is_rev, name):
    """
    lstm cell
    """

    pre_fc_input = fluid.layers.fc(inputs, dim * 4, name = name + "_fc")
    lstm_out, lstm_state = fluid.layers.dynamic_lstm(pre_fc_input, dim * 4,
                                                     is_reverse = is_rev,
                                                     use_peepholes = use_ph,
                                                     name = name + "_lstm")

    return lstm_out, lstm_state


def fc_cell(inputs, dim, act, name):
    """
    fc_cell
    """
    inputs = fluid.layers.fc(inputs, dim, act=act, name=name)

    return inputs


def model(inputs, training = True):
    """
    model
    """
    
    # global tst_input
    # global tst_output
    
    # tst_input = inputs
    wei_idx = 0
    
    for i in range(cfg_lstm_num):
        
        print('lstm', i)
        
        lstm_fw, _ = lstm_cell(inputs, cfg_lstm_dim, cfg_lstm_use_peepholdes, False, "fw_lstm_" + str(i))
        lstm_bw, _ = lstm_cell(inputs, cfg_lstm_dim, cfg_lstm_use_peepholdes, True, "bw_lstm_" + str(i))
        inputs = fluid.layers.concat([lstm_fw, lstm_bw], axis = 1)
        inputs = fc_cell(inputs, cfg_lstm_dim, "relu", "lstm_sf_" + str(i))
            
        if training and (i + 1) != cfg_lstm_num:
            inputs = fluid.layers.dropout(inputs, cfg_dropout_rate)
            
#     print('fc_cell')
#     print(wei_data[wei_idx].shape, wei_data[wei_idx + 1].shape)
#     inputs = fc_cell(inputs, cfg_fc_dim, "relu", "fc", wei_idx)
#     wei_idx += 2
#     print('ctc_fc_cell')
#     print(wei_data[wei_idx].shape, wei_data[wei_idx + 1].shape)
    ctc_outputs = fc_cell(inputs, cfg_ctc_output_dim, None, "ctc_fc")
    # wei_idx += 2
    
    print('mm_lstm')
    nm_output, _ = lstm_cell(inputs, cfg_fc_dim, cfg_lstm_use_peepholdes, True, "nm_lstm")
    nm_output = fluid.layers.sequence_pool(nm_output, "first")
    nm_output = fc_cell(nm_output, cfg_output_dim, None, "nm_fc")
    print('mm_lstm_end')
    
    # tst_output = nm_output
    
    return nm_output, ctc_outputs


def ctc_decode(raw_preds, lod_feats, blank=0):
    """
    简易CTC解码器
    :param text: 待解码数据
    :param blank: 分隔符索引值
    :return: 解码后数据
    """
    sid = 0
    res_ps = []
    for length in lod_feats:
        probs = softmax(raw_preds[sid:sid+length])
        texts = beam_decode(probs, beam_size=6)
        res_p = dict()
        for text, p in texts:
            res = []
            lchar = -1
            for char in text:
                if char != blank and char != lchar:
                    res.append(char)
                lchar = char

            if not res:
                res = [0]

            res = tuple(res)
            res_p[res] = res_p.get(res, 0) + p

        res_p_list = [(k, v) for k, v in res_p.items()]
        res_p_list = sorted(res_p_list, key=lambda x:x[1], reverse=True)
        res_ps.append(res_p_list[0])

        sid += length
    return res_ps


def label_decode(raw_labels, lod_labels):
    decoded = []
    sid = 0
    for length in lod_labels:
        decoded.append(raw_labels[sid:sid+length])
        sid += length
    return decoded


def softmax(x):
    x = np.exp(x)
    s = np.sum(x, axis=1, keepdims=True)
    return x / s

def simple_path_decode(text, blank=0):
    res = []
    lchar = -1
    for char in text:
        if char != blank and char != lchar:
            res.append(char)
        lchar = char
    if not res:
        res = [0]
    return res

def ctc_deocde_single(multi_prob):
    paths = beam_decode(multi_prob, beam_size=6)
    res_text = dict()
    for path, prob in paths:
        text = tuple(simple_path_decode(path))
        res_text[text] = res_text.get(text, 0) + prob
    
    res_text_list = [(k, v) for k, v in res_text.items()] 
    res_text_list = sorted(res_text_list, key=lambda x:x[1], reverse=True) 
    return res_text_list[0]


def decode_and_write_func(sample_idx, label, multi_prob, single_pred):
    
    label_str = ""
    for lid in label:
        label_str += id2hz_map[lid]
    
    #if sample_idx < 36854 or 103758 <=sample_idx< 138519:
    if sample_idx < 36854 or 103758 <=sample_idx< 173282:
        with lock:
            out_fid.write("%d %s %s\n" % (sample_idx, label_str, id2hz_map[single_pred-1]))
            out_fid.flush()
    else:
        multi_pred = np.argmax(multi_prob, -1)
        multi_pred = simple_path_decode(multi_pred)
        #multi_pred, prob = ctc_deocde_single(multi_prob) 
        #if prob < 0.3:
        #    with lock:
        #        out_fid.write("%d %s %s\n" % (sample_idx, label_str, id2hz_map[single_pred-1]))
        #        out_fid.flush()
        #else:
        pred_str = ""
        for pid in multi_pred:
            if pid == 0:
                pred_str += "@"
            else:
                pred_str += id2hz_map[pid-1]
        with lock:
            out_fid.write("%d %s %s\n" % (sample_idx, label_str, pred_str))
            out_fid.flush()


def work_func(wid):
    while True:
        work = queue.get()
        if work is None:
            break
        else:
            decode_and_write_func(*work)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_dir', dest='model_dir')
    args = parser.parse_args()
    
    queue = Queue()
    lock = Lock()

    nb_workers = 4
    workers = []
    
    id2hz_map = dict()
    for idx, line in enumerate(open("hz_utf.txt", encoding="utf8")):
        id2hz_map[idx] = line.strip()
    
    result_file = "logs/output.txt"
    os.system("rm " + result_file)
    out_fid = open(result_file, "a", encoding="utf8")
    
    for wid in range(nb_workers):
        workers.append(Process(target=work_func, args=(wid, )))
    for worker in workers:
        worker.start() 
   
    cfg_ini_model_path = args.model_dir

    inputs = fluid.layers.data(name='input', shape=[None, cfg_input_dim], dtype='float32', lod_level=1)
    labels = fluid.layers.data(name='label', shape=[None, 1], dtype='int32', lod_level=1)
    nm_output, ctc_output = model(inputs, False)

    train_place = fluid.CPUPlace()
    if cfg_use_gpu:
        train_place = fluid.CUDAPlace(0)

    data_load_place = fluid.cpu_places()
    if cfg_use_gpu:
        data_load_place = fluid.cuda_places()

    #train_prg = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(loss_name=loss.name)

    data_loader = fluid.io.DataLoader.from_generator(feed_list=[inputs, labels], capacity=12)
    data_loader.set_batch_generator(batch_generator_creator(), places=data_load_place)

    train_exe = fluid.Executor(train_place)
    train_exe.run(fluid.default_startup_program())

    if cfg_ini_model_path is not None:
        fluid.io.load_params(train_exe, cfg_ini_model_path)

    data_reader.reset()
    batch_num = data_reader.batch_num()
    batch_idx = 0

    t1 = time.time()
    for data in data_loader():

        sample_idx = batch_idx * cfg_batch_size
        if sample_idx < 103680:
            batch_idx += 1
            continue

        raw_labels = np.array(data[0]['label']).flatten()
        # raw_feats = np.array(data[0]['input'])
        lod_labels = np.array(data[0]['label'].recursive_sequence_lengths()).flatten()
        lod_feats = np.array(data[0]['input'].recursive_sequence_lengths()).flatten()

        print("time: %s, sample_idx: %d, sum length: %d queue size: %d"
              % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), sample_idx, np.sum(lod_feats), queue.qsize()))

        t2 = time.time()

        ctc_output_data, tst_output_data = train_exe.run(feed=data, fetch_list=[ctc_output.name, nm_output.name], return_numpy=False)

        raw_preds = np.array(ctc_output_data)
        single_preds = np.array(tst_output_data)

        t3 = time.time()
        label_start = 0
        pred_start = 0
        for idx in range(len(lod_labels)):
            label_len = lod_labels[idx]
            pred_len = lod_feats[idx]
            while queue.qsize() >= 1000:
                time.sleep(3)
            queue.put((idx+sample_idx, 
                       raw_labels[label_start:label_start+label_len],
                       softmax(raw_preds[pred_start:pred_start+pred_len]),
                       np.argmax(single_preds[idx])))
            label_start += label_len
            pred_start += pred_len

        batch_idx += 1
    
    for _ in range(nb_workers):
        queue.put(None)

    for worker in workers:
        worker.join()
