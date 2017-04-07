import tensorflow as tf
import numpy as np
import functools
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import itertools
import metrics as met

mem = Memory("./mycache")


# References
# https://danijar.com/structuring-your-tensorflow-models/
# https://jmetzen.github.io/2015-11-27/vae.html
def xavier_init(fan_in, fan_out, constant=1):
    with tf.name_scope('xavier'):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval=low, maxval=high,
                                 dtype=tf.float32)


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class CmpNN:
    def __init__(self, qdpair, ranking, regu, num_features, num_ranks, training, optimal_rank):
        self.qdpair = qdpair
        self.ranking = ranking
        self.regu = regu
        self.num_features = num_features
        self.num_ranks = num_ranks
        self.training = training
        self.optimal_rank = optimal_rank
        self.prediction
        self.optimize_adam
        self.cost
        self.accuracy
        self.prediction_softmax

    @define_scope
    def prediction(self):
        current_input = self.qdpair
        num_features = self.num_features
        num_ranks = self.num_ranks
        mid = 200

        with tf.name_scope('Model'):
            b0 = tf.Variable(tf.constant(0.1, shape=[int(num_ranks / 2), 1]))
            b1 = tf.reverse(b0, axis=[True, False])
            self.b = tf.squeeze(tf.concat([b0, b1], axis=0))

            w0 = tf.Variable(xavier_init(int(num_features / 2), num_ranks), name='weight')
            w1 = tf.reverse(w0, axis=[True])
            self.W = tf.concat([w0, w1], 0)

            # self.W = tf.Variable(xavier_init(int(num_features), num_ranks))
            # self.b = tf.Variable(tf.constant(0.1, shape=[num_ranks]))
            activ = tf.add(tf.matmul(current_input, self.W), self.b)
            out = activ

            # batch norm
            if self.training is True:
                batch_mean1, batch_var1 = tf.nn.moments(activ, [0])
                z1_hat = (activ - batch_mean1) / tf.sqrt(batch_var1 + 1e-3)
                scale1 = tf.Variable(tf.ones([num_ranks]))
                beta1 = tf.Variable(tf.zeros([num_ranks]))
                BN1 = scale1 * z1_hat + beta1
                out = BN1

        return out

    @define_scope
    def optimize_adam(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        return optimizer.minimize(self.cost)

    @define_scope
    def accuracy(self):
        accur = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prediction_softmax, 1),
                                                tf.argmax(self.ranking, 1)), "float"))
        tf.summary.scalar('accuracy', accur)
        return accur

    @define_scope
    def prediction_softmax(self):
        return tf.nn.softmax(self.prediction)

    @define_scope
    def cost(self):
        #cross_entr = tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.ranking),
        #    name='cross_entropy')
        sq_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.ranking, tf.nn.elu(self.prediction)))))
        regularized_loss = tf.nn.l2_loss(self.W, name='regularization')
        cost = sq_loss + regularized_loss * self.regu
        tf.summary.scalar('error', cost)
        return cost


@mem.cache
def get_vali_data():
    X, y, qid = load_svmlight_file("Fold2/vali.txt",
                                   query_id=True)
    return X, y, qid


@mem.cache
def get_train_data():
    X, y, qid = load_svmlight_file("Fold2/train.txt",
                                   query_id=True)
    return X, y, qid


@mem.cache
def get_test_data():
    X, y, qid = load_svmlight_file("Fold2/test.txt",
                                   query_id=True)
    return X, y, qid


@mem.cache
def data_to_train_on():
    test_x, test_y, test_qid = get_vali_data()
    print('Loaded data')

    train_x = np.zeros(shape=[0, test_x.shape[1] * 2])
    train_y = np.array([[0, 0]])
    for id in list(set(test_qid))[0:70]:
        # print(id)
        index = np.where(test_qid == id)
        train_x = np.concatenate([train_x, getqdpairs(test_x, id, index)], axis=0)
        target_pair, _, _ = gettarget_relations(test_y, id, index)
        train_y = np.concatenate([train_y, target_pair], axis=0)
        # print('Number of qdpairs ', len(train_x))
        # print('Number of rankings ', len(train_y))

    train_x = train_x[1:-1, :]
    train_y = train_y[1:-1, :]
    return train_x, train_y


@mem.cache
def data_to_test():
    test_x, test_y, test_qid = get_test_data()
    print('Loaded data')

    test_data = {}
    for id in list(set(test_qid))[0:20]:
        query_element = {}
        query_element['id'] = id

        index = np.where(test_qid == id)
        train_x = getqdpairs(test_x, id, index)
        query_element['qdpairs'] = train_x

        target_pair, pair_indeces, target_values = gettarget_relations(test_y, id, index)
        query_element['target_pairs'] = target_pair
        query_element['pair_indeces'] = pair_indeces
        query_element['target_vales'] = target_values

        test_data[id] = query_element

    return test_data


def gettarget_relations(test_y, id, index):
    # getting target values corresponding to query id.
    target_values = test_y[index[0]]
    # compute all possible computations
    ypairs = itertools.combinations(target_values, 2)
    ypairs = np.array([np.array([0, 1]) if l < r else np.array([1, 0]) for l, r in ypairs])

    indeces = np.arange(len(target_values))
    qpair_indeces = itertools.combinations(indeces, 2)
    qpair_indeces = np.array([np.array([l, r]) for l, r in qpair_indeces])
    return ypairs, qpair_indeces, target_values


def getqdpairs(test_x, id, index):
    # getting qdpairs corresponding to query id
    qdpair = test_x.todense()[index[0], :]
    # compute all possible computations
    qdpairs = itertools.combinations(qdpair, 2)
    qdpairs = np.array([np.concatenate([np.array(l), np.array(r)], axis=1) for l, r in qdpairs])

    qdpairs = np.squeeze(qdpairs, axis=(1,))
    return qdpairs


def main():
    tf.set_random_seed(1)
    train_x, train_y = data_to_train_on()
    test_data = data_to_test()
    print('Loaded training data')
    num_train_samples = len(train_x)
    # num_test_samples = len(test_x)

    # Placeholders for variables
    qdpair = tf.placeholder(tf.float32, [None, train_x.shape[1]], name='qdpair')
    ranking = tf.placeholder(tf.float32, [None, train_y.shape[1]], name='ranking')
    training = tf.placeholder(tf.bool, None, name='training_phase')
    optimal_ranking = tf.placeholder(tf.float32, [None, train_y.shape[1]], name='optimal_ranking')

    model = CmpNN(qdpair, ranking, regu=0.01,
                  num_features=train_x.shape[1], num_ranks=train_y.shape[1],
                  training=training, optimal_rank=optimal_ranking)

    merged_summary = tf.summary.merge_all()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    logdir = '/tmp/tensorflow_logs/lr/12'
    test_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    num_epochs = 10000

    for epoch in range(num_epochs):
        rand_idx = np.random.randint(num_train_samples, size=int(num_train_samples / num_epochs))
        qdpair_batch = train_x[rand_idx, :]
        ranking_batch = train_y[rand_idx, :]

        _, summary = sess.run([model.optimize_adam, merged_summary],
                              feed_dict={qdpair: qdpair_batch, ranking: ranking_batch, training: True})
        test_writer.add_summary(summary, epoch)

    preds = sess.run([model.prediction_softmax], feed_dict={qdpair: test_data[1]['qdpairs']})

    # COMPUTE NDCG@10 AND MAP
    # iterate through test samples
    ndcgScore = []
    apScore = []
    for k in test_data:
        preds = sess.run([model.prediction_softmax], feed_dict={qdpair: test_data[k]['qdpairs']})
        preds = np.squeeze(np.array(preds))
        scores = np.zeros((int(test_data[k]['pair_indeces'].max())) + 1)
        for idx, row in enumerate(preds):
            scores[test_data[k]['pair_indeces'][idx, np.argmax(row)]] += 1
        # sort is the score for each document by index
        # the arg sort then gets the documents ordered by best to worst relevance wise
        scores = np.argsort(scores)
        # now need to replace the values with the actual relevance score for each document
        for idx, val in enumerate(scores):
            scores[idx] = int(test_data[k]['target_vales'][val])
        if met.NDCG(scores, 10)>0:
            ndcgScore.append(met.NDCG(scores, 10))
        else:
            ndcgScore.append(0)
        apScore.append(met.AP(scores))

    # PRINT NDCG AND MAP FOR TEST SET
    print('NDCG@10 ', np.mean(ndcgScore))
    print('MAP ', np.mean(apScore))


if __name__ == '__main__':
    main()

