import paddle.fluid as fluid
import numpy as np

import model_v4 as pc

def export_model(model_path, output_path):
    
    train_place = fluid.CPUPlace()
    train_exe = fluid.Executor(train_place)
    train_exe.run(fluid.default_startup_program())
    
    fluid.io.load_params(train_exe, model_path)

    with open(output_path, 'w', encoding = 'utf-8') as out_f:
        for block in fluid.default_main_program().blocks:
            for var in block.vars:
                
                if 'tmp' in var:
                    continue
                
                param = fluid.global_scope().find_var(var)
                if param is not None:
                    value = np.array(param.get_tensor())
                    export_stream(var, value, out_f)
                

def export_stream(name, value, out_f):
    
    vshape = value.shape
    value = value.reshape([-1])
    
    out_f.write(name)
    out_f.write('\n')
    
    out_f.write(str(vshape[0]))   
    for ss in vshape[1:]:
        out_f.write(' ')
        out_f.write(str(ss))
    out_f.write('\n')
    
    out_f.write(str(value[0]))
    for v in value[1:]:
        out_f.write(' ')
        out_f.write(str(v))
    out_f.write('\n')
    
  
inputs = fluid.layers.data(name='input', shape=[None, pc.cfg_input_dim], dtype='float32', lod_level=1)
pc.model(inputs, False)
#export_model('D:\\_HW\\fuxian_model\\20', 'D:\\_HW\\fuxian_model\\20.txt')
export_model("./ctc_model_paddle/43", "./ctc_model_txt/43.txt")
#export_model('D:\\_HW\\tmp_model\\12', 'D:\\_HW\\model_v4_1\\tmp.txt')
        
#train()
