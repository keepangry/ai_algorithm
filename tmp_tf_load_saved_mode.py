#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 10:41 AM
# @Author  : 0@keepangry.com
# @Site    : 
# @File    : pred_example.py
# @Software: PyCharm



# 从导出目录中加载模型，并生成预测函数。
# https://tensorflow.google.cn/api_docs/python/tf/contrib/predictor/from_saved_model

import tensorflow as tf
import numpy as np
import os

signature_key = 'predict_images'
input_key = 'images'
output_key = 'scores'

## tf.app.flags.DEFINE_string("valid_data", "viewfs://hadoop-meituan/user/hadoop-waimaircmining/yangsen07/deeplearning/demo/order_seq/saved_model/1", "valid_data")
## print(tf.app.flags.FLAGS.valid_data)
## os.popen("hadoop fs -get {}/part*00000 valid_data".format(saved_model_hdfs))
## hope dfs -get viewfs://hadoop-meituan/user/hadoop-waimaircmining/yangsen07/deeplearning/demo/order_seq/saved_model/1/* valid_data

# export_dir = "logs/saved_model/1"
saved_model_hdfs = "viewfs://hadoop-meituan/user/hadoop-waimaircmining/yangsen07/deeplearning/demo/order_seq/saved_model/1"
export_dir = saved_model_hdfs

"""
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x},
            outputs={'scores': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={'predict_images': prediction_signature},
        legacy_init_op=legacy_init_op)

    builder.save()
 """




_x = np.zeros([128, 24, 35])

with tf.Session(graph=tf.Graph()) as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    signature = meta_graph_def.signature_def

    # 从signature中找出具体输入输出的tensor name
    x_tensor_name = signature[signature_key].inputs[input_key].name
    y_tensor_name = signature[signature_key].outputs[output_key].name

    # 获取tensor 并inference
    x = sess.graph.get_tensor_by_name(x_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)

    # _x 实际输入待inference的data
    result = sess.run(y, feed_dict={x: _x})
    print(result)
