import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from keras import backend as K, Model


def export_model(model: Model):
    sess = tf.Session()

    K.set_session(sess)
    K.set_learning_phase(0)

    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        {"input": model.input},
        {"output": model.output}
    )

    builder = saved_model_builder.SavedModelBuilder('servable/main')

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess.run(init_op)

    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature,
        },
        legacy_init_op=legacy_init_op
    )

    builder.save()
