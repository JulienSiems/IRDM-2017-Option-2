import tensorflow as tf
import numpy as np
import functools
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import metrics as met
mem = Memory("./mycache")

# References
# https://danijar.com/structuring-your-tensorflow-models/
# https://jmetzen.github.io/2015-11-27/vae.html
def xavier_init(fan_in, fan_out, constant = 1):
    with tf.name_scope('xavier'):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval = low, maxval = high,
                                 dtype = tf.float32)

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

class LogisticRegression:

    def __init__(self, qdpair, ranking, regu, num_features, num_ranks, training):
        self.qdpair = qdpair
        self.ranking = ranking
        self.regu = regu
        self.num_features = num_features
        self.num_ranks = num_ranks
        self.training = training
        self.prediction
        self.optimize
        self.cost
        self.accuracy
        self.prediction_softmax

    @define_scope
    def prediction(self):
        current_input = self.qdpair
        num_features = self.num_features
        num_ranks = self.num_ranks

        # ENCODER
        with tf.name_scope('Model'):
            self.W = tf.Variable(xavier_init(num_features, num_ranks), name = 'weight')
            self.b = tf.Variable(tf.zeros(num_ranks), name = 'bias')
            activ = tf.add(tf.matmul(current_input, self.W), self.b)
            out = activ
            if self.training is True:
                batch_mean1, batch_var1 = tf.nn.moments(activ, [0])
                z1_hat = (activ - batch_mean1) / tf.sqrt(batch_var1 + 1e-3)
                scale1 = tf.Variable(tf.ones([num_ranks]))
                beta1 = tf.Variable(tf.zeros([num_ranks]))
                BN1 = scale1*z1_hat + beta1
                out = BN1
        return out

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        return optimizer.minimize(self.cost)

    @define_scope
    def accuracy(self):
        accur = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.ranking, 1)), "float"))
        tf.summary.scalar('accuracy', accur)
        return accur

    @define_scope
    def prediction_softmax(self):
        return tf.nn.softmax(self.prediction)

    @define_scope
    def cost(self):
        cross_entr = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits = self.prediction, labels = self.ranking),
            name = 'cross_entropy')
        regularized_loss = tf.nn.l2_loss(self.W, name='regularization')
        cost = cross_entr + regularized_loss*self.regu
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
def data_to_test():
    test_x, test_y, test_qid = get_test_data()
    print('Loaded Test data')
    test_x = test_x.todense()
    test_data = {}
    for id in list(set(test_qid))[0:20]:
        query_element = {}
        query_element['id'] = id
        index = np.where(test_qid == id)

        query_element['qd'] = test_x[index[0], :]
        query_element['target'] = test_y[index[0]]
        test_data[id] = query_element

    return test_data, test_x, test_y


def main():
    # Import training data
    train_x, train_y, qid = get_train_data()
    train_x = train_x.todense()
    train_y = pd.DataFrame(train_y, columns=['relevance'])
    train_y[train_y.columns[0]] = train_y[train_y.columns[0]].map({0: '0', 1: '1', 2: '2', 3: '3', 4: '4'})
    train_y = pd.get_dummies(train_y).as_matrix()

    num_samples = train_x.shape[0]
    print('Loaded training data')

    # Import test data
    test, test_x, test_y = data_to_test()
    test_y = pd.DataFrame(test_y, columns=['relevance'])
    test_y[test_y.columns[0]] = test_y[test_y.columns[0]].map({0: '0', 1: '1', 2: '2', 3: '3', 4: '4'})
    test_y = pd.get_dummies(test_y).as_matrix()
    print('Loaded test data')

    # Placeholders for variables
    qdpair = tf.placeholder(tf.float32, [None, train_x.shape[1]], name='qdpair')
    ranking = tf.placeholder(tf.float32, [None, train_y.shape[1]], name='qdpair')
    training = tf.placeholder(tf.bool, None, name='training_phase')
    optimal_ranking = tf.placeholder(tf.float32, [None, train_y.shape[1]], name='optimal_ranking')

    model = LogisticRegression(qdpair, ranking, regu=0,
                               num_features=train_x.shape[1], num_ranks=train_y.shape[1],
                               training=training)

    merged_summary = tf.summary.merge_all()
    iteration = str(8)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    training_writer = tf.summary.FileWriter('/tmp/tensorflow_logs/lr/' + iteration + '_train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter('/tmp/tensorflow_logs/lr/' + iteration + '_test', graph=tf.get_default_graph())

    num_epochs = 30000
    for epoch in range(num_epochs):
        rand_idx = np.random.randint(num_samples, size=int(num_samples/num_epochs))
        qdpair_batch = train_x[rand_idx, :]
        ranking_batch = train_y[rand_idx, :]
        _, summary = sess.run([model.optimize, merged_summary],
                              feed_dict={qdpair: qdpair_batch, ranking: ranking_batch, training: True})
        training_writer.add_summary(summary, epoch)
        if epoch % 100 == 0:
            _, summary = sess.run([model.cost, merged_summary],
                               feed_dict={qdpair: test_x, ranking: test_y, training: True})
            test_writer.add_summary(summary, epoch)

    ndcgScore = []
    apScore = []
    for k in test:
        preds = np.squeeze(sess.run([model.prediction_softmax],
                                    feed_dict={qdpair: test[k]['qd'], training: True}))
        ranks = [np.argmax(pred) for pred in preds]
        # sort is the score for each document by index
        # the arg sort then gets the documents ordered by best to worst relevance wise
        scores = np.argsort(ranks)
        # now need to replace the values with the actual relevance score for each document
        for idx, val in enumerate(scores):
            scores[idx] = int(test[k]['target'][val])
        if met.NDCG(scores, 10) > 0:
            ndcgScore.append(met.NDCG(scores, 10))
            apScore.append(met.AP(scores))
    print(np.mean(ndcgScore))
    print(np.mean(apScore))

if __name__ == '__main__':
    main()
