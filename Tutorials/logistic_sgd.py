import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow = True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow = True).shape[0] / batch_size
    
    print "... building the model"
    
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    
    classifier = LogisticRegression(input =x, n_in= 28*28, n_out = 10)
    cost = classifier.negative_log_likelihood(y)
    
    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
        x: test_set_x[index * batch_size: (index+1) * batch_size],
        y: test_set_y[index * batch_size: (index+1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
        x: valid_set_x[index * batch_size: (index+1) * batch_size],
        y: valid_set_y[index * batch_size: (index+1) * batch_size]
        }
    )
        
    g_W = T.grad(cost = cost, wrt = classifier.W)
    g_b = T.grad(cost = cost, wrt = classifier.b)

    
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
              (classifier.b, classifier.b - learning_rate * g_b)]
    
    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_set_x[index * batch_size: (index+1) * batch_size],
            y: train_set_y[index * batch_size: (index+1) * batch_size]
        }
    )
    
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience /2 )
    best_validation_loss = numpy.inf
    test_score = 0
    start_time = time.clock()
    done_looping = False
    epoch = 0
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch -1 ) * n_train_batches + minibatch_index
            
            if (iter +1) % validation_frequency == 0:
                curr_validation_loss = numpy.mean([validate_model(i) for i in xrange(n_valid_batches)])
                print ( 'epoch %i, minibatch %i/%i, validation error %f %%'%(epoch, minibatch_index+1, n_train_batches, curr_validation_loss*100.))
                
                if curr_validation_loss < best_validation_loss:
                    
                    if curr_validation_loss < best_validation_loss* improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    
                    best_validation_loss = curr_validation_loss
                    test_score = numpy.mean([test_model(i) for i in xrange(n_test_batches)])
                    print ( 'epoch %i, minibatch %i/%i, validation error %f %%'%(epoch, minibatch_index+1, n_train_batches, test_score*100.))
                
            if patience <= iter:
                done_looping = True
                break
        end_time = time.clock()
        print (
            (
                'Optimization complete with best validation score of %f %%,' 
                 'with performance of %f %%'
            ) 
            %(best_validation_loss * 100., test_score * 100.)
        )
        print (
            ('The code run for %d epochs with %f epochs/sec ')
            %
            (epoch, 1. * epoch /(end_time - start_time) )
            
            )  
   
if __name__ == '__main__':
    sgd_optimization_mnist()