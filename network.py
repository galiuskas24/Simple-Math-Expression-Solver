import tensorflow as tf

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 48, 48, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[7, 7],
        padding="same",
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[7, 7],
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[7, 7],
        padding="same",
        activation=tf.nn.relu
    )

    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=2
    )

    pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 128])
    dense = tf.layers.dense(inputs=pool3_flat, units=2048, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # units = number of symbols
    logits = tf.layers.dense(inputs=dropout, units=nof_labels)
    #logits = tf.layers.dense(inputs=dropout, units=len(labels))

    predictions = {
        # generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=nof_labels)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)