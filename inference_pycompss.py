"""#"""

import os
import argparse
import tensorflow as tf
from pycompss.api.task import task
from pycompss.api.parameter import OUT

NUM_CORES = 1
DISABLE_OPTIMIZATION = True

def load_graph(frozen_graph_filename):
  """#"""
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
      tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name="",
        op_dict=None,
        producer_op_list=None
      )
  return graph

@task(res=OUT, returns=int)
def main(path, imagepath, img, res):
  image = str(imagepath + img)
  modelpath = str(path + 'models/frozen_model.pb')
  config = tf.ConfigProto(
      inter_op_parallelism_threads=NUM_CORES,
      intra_op_parallelism_threads=NUM_CORES
      )

  if DISABLE_OPTIMIZATION:
      config.graph_options.optimizer_options.opt_level = \
        config.graph_options.optimizer_options.L0

  graph = load_graph(modelpath)
  x = graph.get_tensor_by_name('image_path:0')
  y = graph.get_tensor_by_name('output:0')

  with tf.Session(graph=graph, config=config) as sess:
      res = sess.run(tf.squeeze(y), feed_dict={x: image})

  return res


if __name__ == '__main__':
  from pycompss.api.api import compss_wait_on
  cwd = os.getcwd()
  path = cwd + '/test_tf/'
  imagepath = path + 'input/'
  results = []
  filenames = []
  for filename in os.listdir(imagepath):
      filenames.append(filename)
      results.append(main(str(path), str(imagepath), str(filename), 1))

  results = compss_wait_on(results)

  for ind in range(0, len(filenames)):  
      print("result of " + str(filenames[ind]) + ": " + str(results[ind]))
