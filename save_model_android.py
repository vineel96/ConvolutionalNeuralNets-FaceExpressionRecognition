
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

def freeze_graph(sess):

    tf.train.write_graph(sess.graph_def,'.','FER_Library1.pbtxt')

    #Save to Local FileSystem
    saver1=tf.train.Saver(tf.all_variables())
    saver1.save(sess,"./FER_Library1.ckpt",global_step=0)

    #Freeze the Graph
    input_graph_path='FER_Library1.pbtxt'
    checkpoint_path='./FER_Library1.ckpt'
    input_saver_def_path=""
    input_binary=False
    output_node_names="final_result"
    restore_op_name="save/restore_all"
    filename_tensor_name="save/Const:0"
    output_frozen_graph_name="frozen_"+"FER_Library1.pb"
    output_optimized_graph_name = "optimized_" + "FER_Library1.pb"
    clear_devices=True
    freeze_graph.freeze_graph(input_graph_path,input_saver_def_path,input_binary,checkpoint_path,
                              output_node_names,restore_op_name,filename_tensor_name,output_frozen_graph_name,
                              clear_devices,"")

    #Optimize for inference
    input_graph_def=tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name,"rb") as f:
        data=f.read()
        input_graph_def.ParseFromString(data)
    output_graph_def=optimize_for_inference_lib.optimize_for_inference(input_graph_def,
                                                                       ["Input"],
                                                                       ["final_result"],
                                                                       tf.float32.as_datatype_enum)

    #Save the optimized graph
    f=tf.gfile.FastGFile(output_optimized_graph_name,"w")
    f.write(output_graph_def.SerializeToString())



    #Visualize model using graph
    writer = tf.summary.FileWriter("./CNNFlow")
    writer.add_graph(sess.graph)