"""
paddle.1.8.2
"""

import os
import platform
import paddle.fluid as fluid
import paddle_bd as ds
import numpy as np
import datetime
import sys
import paddle
paddle.enable_static()

cfg_use_gpu = False

cfg_lr = 0.0001
cfg_dropout_rate = 0.1

cfg_lstm_dim = 128
cfg_lstm_num = 4
cfg_lstm_use_peepholdes = False

cfg_fc_dim = 128
cfg_input_dim = 8
cfg_nm_output_dim = 28800
cfg_ctc_output_dim = 6400

cfg_batch_size = 512
cfg_max_epoch = 900

cur_model_idx = int(sys.argv[1])
cfg_model_path = './output_lr0.0001'
cfg_ini_model_path = './output/%d' % cur_model_idx

tst_input = None
tst_output = None

if (platform.system()[0] == 'W'):
    cfg_batch_size = 32
    cfg_wei_code_path = 'D:\\_HW\\model_v4_1\\28.txt'
    cfg_hz_feat_path = 'D:\\_HW\\hw_dz_data.bin'
    cfg_model_path = 'D:\\_HW\\tmp_model\\'
    cfg_use_gpu = False

    os.environ['CPU_NUM'] = '2'
else:
    cfg_wei_code_path = os.path.join('./ctc_model_txt', '43.txt')
    cfg_hz_feat_path = os.path.join('./train_data', 'hw_dz_data.bin')

train_place = None
data_reader = ds.Dataset()
data_reader.open(None, cfg_hz_feat_path, cfg_batch_size)

wei_data = data_reader.read_weight_data(cfg_wei_code_path)


def batch_generator_creator():
    """
    read batch data
    """

    def __reader__():
        label, lod_label, feat, lod_feat = data_reader.next()
        while label is not None:

            label = label.astype(np.int64)

            iinputs = fluid.create_lod_tensor(feat, [lod_feat], train_place)
            ilabels = fluid.create_lod_tensor(label, [lod_label], train_place)
            # print(ilabels)
            #print(ilabels.shape, ilabels.dtype)

            yield iinputs, ilabels

            label, lod_label, feat, lod_feat = data_reader.next()

    return __reader__


def lstm_cell(inputs, dim, use_ph, is_rev, name, wei_idx):
    """
    lstm cell
    """

    if wei_idx == -1:
        pre_fc_input = fluid.layers.fc(inputs, dim * 4, name=name + "_fc")
        lstm_out, lstm_state = fluid.layers.dynamic_lstm(pre_fc_input, dim * 4,
                                                         is_reverse=is_rev,
                                                         use_peepholes=use_ph,
                                                         name=name + "_lstm")
    else:

        print(wei_data[wei_idx].shape, wei_data[wei_idx + 1].shape,
              wei_data[wei_idx + 2].shape, wei_data[wei_idx + 3].shape)

        fc_k = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(
            wei_data[wei_idx]), trainable=False)
        fc_b = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(
            wei_data[wei_idx + 1]), trainable=False)
        lstm_k = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(
            wei_data[wei_idx + 2]), trainable=False)
        lstm_b = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(
            wei_data[wei_idx + 3]), trainable=False)

        pre_fc_input = fluid.layers.fc(
            inputs, dim * 4, param_attr=fc_k, bias_attr=fc_b, name=name + "_fc")
        lstm_out, lstm_state = fluid.layers.dynamic_lstm(pre_fc_input, dim * 4,
                                                         param_attr=lstm_k,
                                                         bias_attr=lstm_b,
                                                         is_reverse=is_rev,
                                                         use_peepholes=use_ph,
                                                         name=name + "_lstm")
    return lstm_out, lstm_state


def fc_cell(inputs, dim, act, name, wei_idx):
    """
    fc_cell
    """

    if wei_idx == -1:
        inputs = fluid.layers.fc(inputs, dim, act=act, name=name)
    else:
        fc_k = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(
            wei_data[wei_idx]), trainable=False)
        fc_b = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(
            wei_data[wei_idx + 1]), trainable=False)
        inputs = fluid.layers.fc(inputs, dim, act=act,
                                 param_attr=fc_k, bias_attr=fc_b, name=name)

    return inputs


def model(inputs, training=True):
    """
    model
    """

    global tst_input
    global tst_output

    tst_input = inputs
    wei_idx = 0

    for i in range(cfg_lstm_num):

        print('lstm', i)

        lstm_fw, _ = lstm_cell(
            inputs, cfg_lstm_dim, cfg_lstm_use_peepholdes, False, "fw_lstm_" + str(i), wei_idx)
        wei_idx += 4
        lstm_bw, _ = lstm_cell(
            inputs, cfg_lstm_dim, cfg_lstm_use_peepholdes, True, "bw_lstm_" + str(i), wei_idx)
        wei_idx += 4
        inputs = fluid.layers.concat([lstm_fw, lstm_bw], axis=1)
        inputs = fc_cell(inputs, cfg_lstm_dim, "relu",
                         "lstm_sf_" + str(i), wei_idx)
        wei_idx += 2

        if training and (i + 1) != cfg_lstm_num:
            inputs = fluid.layers.dropout(inputs, cfg_dropout_rate)

#     print('fc_cell')
#     print(wei_data[wei_idx].shape, wei_data[wei_idx + 1].shape)
#     inputs = fc_cell(inputs, cfg_fc_dim, "relu", "fc", wei_idx)
#     wei_idx += 2
    print('ctc_fc_cell')
    print(wei_data[wei_idx].shape, wei_data[wei_idx + 1].shape)
    ctc_outputs = fc_cell(inputs, cfg_ctc_output_dim, None, "ctc_fc", wei_idx)
    wei_idx += 2

    print('mm_lstm')
    nm_output, _ = lstm_cell(
        inputs, cfg_fc_dim, cfg_lstm_use_peepholdes, True, "nm_lstm", -1)
    nm_output = fluid.layers.sequence_pool(nm_output, "first")
    nm_output = fc_cell(nm_output, cfg_nm_output_dim, None, "nm_fc", -1)
    print('mm_lstm_end')

    tst_output = nm_output

    return nm_output, _


def train_ctc():
    """
    train ctc
    """

    global train_place
    inputs = fluid.layers.data(
        name='input', shape=[None, cfg_input_dim], dtype='float32', lod_level=1)
    labels = fluid.layers.data(
        name='label', shape=[None, 1], dtype='int64', lod_level=1)

    nm_output, _ = model(inputs, True)

    ctc_loss = fluid.layers.softmax_with_cross_entropy(nm_output, label=labels)
    loss = ctc_loss

    loss = fluid.layers.reduce_mean(loss)

    optimizer = fluid.optimizer.Adam(
        cfg_lr, grad_clip=fluid.clip.GradientClipByValue(min=-0.5, max=0.5))
    optimizer.minimize(loss)

    train_place = fluid.CPUPlace()
    if cfg_use_gpu:
        train_place = fluid.CUDAPlace(0)

    data_load_place = fluid.cpu_places()
    if cfg_use_gpu:
        data_load_place = fluid.cuda_places()

    train_prg = fluid.CompiledProgram(
        fluid.default_main_program()).with_data_parallel(loss_name=loss.name)

    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=[inputs, labels], capacity=12)
    data_loader.set_batch_generator(
        batch_generator_creator(), places=data_load_place)

    train_exe = fluid.Executor(train_place)
    train_exe.run(fluid.default_startup_program())

    if cfg_ini_model_path is not None:
        fluid.io.load_params(train_exe, cfg_ini_model_path)

    for epoch in range(cur_model_idx + 1, cfg_max_epoch):
        data_reader.reset(True)
        batch_num = data_reader.batch_num()
        batch_idx = 0
        t = datetime.datetime.now()
        for data in data_loader():

            nt1 = datetime.datetime.now()
            loss_data = train_exe.run(
                train_prg, feed=data, fetch_list=[loss.name])
            nt2 = datetime.datetime.now()
            print(nt2 - nt1, nt2 - t)
            print(epoch, batch_idx, batch_num, loss_data)
            t = nt2
            batch_idx += 1

        fluid.io.save_params(train_exe, os.path.join(
            cfg_model_path, str(epoch)))


def save_infer_model(in_path, out_path):
    """
    save inference model
    """

    inputs = fluid.layers.data(
        name='input', shape=[None, cfg_input_dim], dtype='float32', lod_level=1)
    _, ctc_output = model(inputs, False)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.io.load_params(exe, in_path)
    fluid.io.save_inference_model(out_path, ['input'], [ctc_output], exe)


def lod2file(path, lod_tensor):
    """
    save lod-tensor
    """

    lod = lod_tensor.lod()[0]
    data = np.array(lod_tensor)

    print(type(lod), len(lod))

    with open(path, 'w', encoding='utf-8') as out_f:
        for i in range(len(lod) - 1):
            b = lod[i]
            e = lod[i + 1]

            lenh = e - b
            out_f.write(str(lenh))
            out_f.write(' 1 1 ')
            out_f.write(str(data.shape[1]))
            out_f.write('\n')

            ndata = data[b:e, :]
            ndata = ndata.reshape([-1])
            out_f.write(str(ndata[0]))
            for nd in ndata[1:]:
                out_f.write(' ')
                out_f.write(str(nd))
            out_f.write('\n')


def test_io(model_path, in_path, out_path):
    """
    test io
    """

    global train_place
    global tst_input
    global tst_output

    inputs = fluid.layers.data(
        name='input', shape=[None, cfg_input_dim], dtype='float32', lod_level=1)
    labels = fluid.layers.data(
        name='label', shape=[None, 1], dtype='int64', lod_level=1)

    _, ctc_output = model(inputs, False)

    ctc_loss = fluid.layers.warpctc(input=ctc_output, label=labels)
    loss = ctc_loss

    loss = fluid.layers.reduce_mean(loss)

    optimizer = fluid.optimizer.Adam(
        cfg_lr, grad_clip=fluid.clip.GradientClipByValue(min=-0.5, max=0.5))
    optimizer.minimize(loss)

    train_prg = fluid.CompiledProgram(
        fluid.default_main_program()).with_data_parallel(loss_name=loss.name)
    train_place = fluid.CPUPlace()
    if cfg_use_gpu:
        train_place = fluid.CUDAPlace(0)

    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=[inputs, labels], capacity=10, use_multiprocess=True)
    data_loader.set_batch_generator(
        batch_generator_creator(), places=train_place)

    train_exe = fluid.Executor(train_place)
    train_exe.run(fluid.default_startup_program())

    fluid.io.load_params(train_exe, model_path)

    data_reader.reset(True)
    batch_num = data_reader.batch_num()
    batch_idx = 0
    print(batch_num)
    for data in data_loader():
        ti, to = train_exe.run(train_prg, feed=data, fetch_list=[
                               tst_input.name, tst_output.name], return_numpy=False)
        lod2file(in_path, ti)
        lod2file(out_path, to)
        batch_idx += 1
        break


train_ctc()
#test_io('D:\\new_hw\\danzi\\model\\8', 'd:\\in.txt', 'd:\\out.txt')
