import tensorflow as tf
import numpy as np

def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the batch size and number of batches we can make
    batch_size = n_seqs * n_steps
    n_batches = len(arr)//batch_size
    
    # Keep only enough characters to make full batches
    arr = np.array(arr[:batch_size * n_batches])
    
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, n_batches * n_steps))
    
    batches = []
    
    #35, 5
    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:,n:n + n_steps]

        #Shapes:  arr -> (128, 35) | x -> (128, 5)
        
        # Initialise y, the same size as x and with zeros
        y = x * 0
        
        #The last step we need to process the batch different
        if n == (arr.shape[1] - n_steps):
            #X will only return four columns, first column of arr
            y[:, :-1], y[:, -1] = x[:, 1:], np.roll(arr[:, 0],-1) 
        else:
            # The targets, shifted by one
            y = arr[:,n + 1:n + n_steps + 1]
            
        batches.append([x, y])
        
    batches = np.array(batches)
    #print(batches.shape)
    
    return batches

def reset_graph():
    tf.reset_default_graph()

def get_inputs():
    """
    Creates TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs_ = tf.placeholder(tf.int32, [None, None], name="input")
    targets_ = tf.placeholder(tf.int32, [None, None], name='targets')
    lr_ = tf.placeholder(tf.float32, name='lr')
    
    return inputs_, targets_, lr_

def get_init_cell(batch_size, rnn_layers, rnn_size, keep_prob):
    """
    Creates an RNN Cell and initialises it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    print("RNN Layers: {} and Size: {}, Batch Size: {}".format(rnn_layers, rnn_size, batch_size))

    #For Tensorflow 1.0.0
    
    # Build the LSTM Cell
    #lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
    
    # Add dropout to the cell outputs to prevent overfitting
    #drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    #cell = tf.contrib.rnn.MultiRNNCell([lstm] * rnn_layers)

    ###Unhash above function if you are using TensorFlow Version: 1.0.0

    #For Tensorflow 1.0.0

    def lstm():
        lst = tf.contrib.rnn.LSTMCell(rnn_size, reuse=tf.get_variable_scope().reuse)
        return tf.contrib.rnn.DropoutWrapper(lst, output_keep_prob=keep_prob)


    cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(rnn_layers)], state_is_tuple = True)

    ###Unhash above function if you are using TensorFlow Version: 1.1.0
    
    initial_state = cell.zero_state(batch_size, tf.float32)
    #initial_state = cell.zero_state(batch_size, tf.int32)
    #print(initial_state)
    initial_state = tf.identity(initial_state, name="initial_state")
    #print(initial_state)  
    
    return cell, initial_state

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim)), name="embedding")
    embed = tf.nn.embedding_lookup(embedding, input_data)
    
    print(embed.get_shape())
    
    return embed

def build_rnn(cell, inputs, batch_size, rnn_layers, rnn_size, keep_prob):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :parm batch_size: Number of training examples
    :param rnn_size: Size of rnns
    :param rnn_layers: Number of rnn layers
    :param keep_prob: Dropout
    :return: Tuple (Outputs, Final State)
    """
    
    #print("Input:", inputs)
    cell, initial_state = get_init_cell(batch_size, rnn_layers, rnn_size, keep_prob)
    
    #dynamic_rnn returns 
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype="float32")
    
    final_state = tf.identity(final_state, name="final_state")
    #print(final_state)
    
    return outputs, final_state

def build_nn(cell, input_data, vocab_size, embed_dim, batch_size, rnn_layers, rnn_size, keep_prob):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :parm batch_size: Number of training examples
    :param rnn_size: Size of rnns
    :param rnn_layers: Number of rnn layers
    :param keep_prob: Dropout
    :return: Tuple (Logits, FinalState)
    """
    inputs = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, inputs, batch_size, rnn_layers, rnn_size, keep_prob)
    
    #activation_fn=None to maintain a Linear activation - https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    
    return logits, final_state

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    tensor_names = ["input:0", "initial_state:0", "final_state:0", "probs:0"]
    
    return tuple(loaded_graph.get_tensor_by_name(tn) for tn in tensor_names)
