# Job scheduler of CNN workers with PyCOMPSs

The following code implements an inference process of multiple parallel images
using PyCOMPSs on top of TensorFlow.

The trained model weights have been exported using `tf.train.export_meta_graph`
[[1]][API][[2]][howto], which returns a single file with the model definition
(layer configuration, layer shape, hyperparameters, etc.) and the model weights.
Then the model weights have been *frozen* , which means that instead of having
them defined as variables, and initializing the graph every time the model is
loaded, we save them as constants in order to skip the initialization step.

As we use this model for inference, we do not need to further train the weights,
thus we can save them as constants. This method is often used to reduce the
overhead of the network initialization, and we could prune the execution graph by
removing the unused nodes that are only used for the training phase, although it
is out of the scope for this test.

The following piece of code shows how each of the workers will load the model
using the `tf.import_graph_def`[[1]][API2] function. We load the model weights
and the graph structure from the protobuf file:

```python
def load_graph(frozen_graph_filename):
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
          producer_op_list=None)
  return graph
```

We define the main function the workers will run. The input of the model is an
image path (string) and the output is the predicted class (integer). Each one of
the tasks will get the image path to process, and will be loaded to the model
using the *feed_dict* parameter. We constrain the resources each one of the
workers is able to use using `tf.ConfigProto`. For this specific test we were
interested to constrain each worker to use a single thread.

```python
NUM_CORES = 1
DISABLE_OPTIMIZATION = True

@task(res=OUT, returns=int)
def main(path, imagepath, img, res):
    image = str(imagepath + img)
    modelpath = str(path + 'models/frozen_model.pb')
    config = tf.ConfigProto(
        inter_op_parallelism_threads=NUM_CORES,
        intra_op_parallelism_threads=NUM_CORES)

    if DISABLE_OPTIMIZATION:
        config.graph_options.optimizer_options.opt_level = \
            config.graph_options.optimizer_options.L0

    graph = load_graph(modelpath)
    x = graph.get_tensor_by_name('image_path:0')
    y = graph.get_tensor_by_name('output:0')

    with tf.Session(graph=graph, config=config) as sess:
        res = sess.run(tf.squeeze(y), feed_dict={x: image})

    return res
```

The loop to initialize the workers and gather the results is the following:

```python
import os
import argparse
import tensorflow as tf
from pycompss.api.task import task
from pycompss.api.parameter import OUT
from pycompss.api.api import compss_wait_on

cwd = os.getcwd()
path = cwd + '/test_tf/'
imagepath = path + 'input/'
results = filenames = []
for filename in os.listdir(imagepath):
    filenames.append(filename)
    results.append(main(str(path), str(imagepath), str(filename), 1))

results = compss_wait_on(results)

for ind in range(0, len(filenames)):
    print("result of " + str(filenames[ind]) + ": " + str(results[ind]))
```
We have chosen to use COMP Superscalar (COMPSs), a framework which aims to ease the development and execution of applications for distributed infrastructures. The specific chosen framework is PyCOMPSs[[4]][COMPSs] (the COMPSs Python binding). COMPSs is also complemented with a set of tools for facilitating the development, execution monitoring and post-mortem performance analysis. One of the tools is Extrae, a package devoted to generate Paraver trace-files, to monitor the correct execution of our application. With COMPSs, we can also obtain a graph of the tasks execution, to check the correct execution of the parallel application.
To parallelize our task using PyCOMPSs, only a few lines of code need to be added:
- Import the pycompss task to use the parallel task decorators, the compss_wait_on to wait for all the parallel tasks to finish, and the needed parameters for the task decorators.
```python
from pycompss.api.task import task
from pycompss.api.parameter import OUT
from pycompss.api.api import compss_wait_on
```
- To add a decorator to the tasks we want to be parallel, and insert the needed information about the parameters. More information on this can be found in the COMPSs manual [[5]][COMPSsManual].
```python
@task(res=OUT, returns=int)
```
[API]: https://www.tensorflow.org/api_docs/python/state_ops/exporting_and_importing_meta_graphs#export_meta_graph
[API2]: https://www.tensorflow.org/api_docs/python/framework/utility_functions#import_graph_def
[howto]: https://www.tensorflow.org/how_tos/meta_graph/
[COMPSs]: http://journals.sagepub.com/doi/abs/10.1177/1094342015594678
[COMPSsManual]: http://compss.bsc.es/releases/compss/latest/docs/COMPSs_User_Manual_App_Development.pdf?tracked=true
