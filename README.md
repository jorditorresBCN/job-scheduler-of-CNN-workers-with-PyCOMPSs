# Job scheduler of CNN workers with PyCOMPSs

The following code implements an inference process of multiple parallel images
using PyCOMPSs on top of TensorFlow.

## TensorFlow inference

The trained model weights have been exported using `tf.train.export_meta_graph`
[[1]][API][[2]][howto], which returns a single file with the model definition
(layer configuration, shape, hyperparameters, etc.) and the model weights.
Then, we have *frozen* the model weights, which means that instead of having
them defined as variables, and initializing the graph every time the model is
loaded, we save them as constants in order to skip the initialization step.

The following piece of code shows how each of the workers will load the model
using the `tf.import_graph_def`[[1]][API2] function:
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

We then define the main function for the workers to run. The input of the model
is an image path (string) and the output is the predicted class (integer).

```python
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








[API]: https://www.tensorflow.org/api_docs/python/state_ops/exporting_and_importing_meta_graphs#export_meta_graph
[API2]: https://www.tensorflow.org/api_docs/python/framework/utility_functions#import_graph_def
[howto]: https://www.tensorflow.org/how_tos/meta_graph/
