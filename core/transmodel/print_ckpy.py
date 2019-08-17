from tensorflow.python import pywrap_tensorflow
import os

checkpoint_path=os.path.join('./model.ckpt')
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()
for key in sorted(var_to_shape_map):
    temp = reader.get_tensor(key)
    print('tensor_name: ',key)
