#!/usr/bin/env python
__author__ = 'solivr'

import tensorflow as tf


class PredictionModel:

    def __init__(self, model_dir, session=None, graph=None):
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()
        if graph:
            self.graph = graph
        else:
            self.graph = tf.get_default_graph()
        self.model = tf.saved_model.loader.load(self.session, ['serve'], model_dir)

        self._input_dict, self._output_dict = _signature_def_to_tensors(self.model.signature_def['predictions'],
                                                                        self.graph)

    def predict(self, image):
        output = self._output_dict
        return self.session.run(output, feed_dict={self._input_dict['images']: image})


def _signature_def_to_tensors(signature_def, g):  # from SeguinBe
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}
