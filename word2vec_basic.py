from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import pickle
from time import time


def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename


def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    full_data = []
    for i in range(1): 
      data = f.read(f.namelist()[i]).replace(b'\x81', b"").replace(b'\x8d', b"").replace(b'\x8f', b"")
      data = data.replace(b'\x90', b"").replace(b'\x9d', b"").decode('windows-1252')
      full_data.extend(tf.compat.as_str(data).split())
      print(f.namelist()[i])  
  return full_data


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


def generate_batch_SG(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


def generate_batch_CBOW(batch_size, bag_window):
  global data_index
  span = 2 * bag_window + 1 # [ bag_window target bag_window ]
  batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size):
    # just for testing
    buffer_list = list(buffer)
    labels[i, 0] = buffer_list.pop(bag_window)
    batch[i] = buffer_list
    # iterate to the next buffer
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels


def plot_with_labels(two_dim_embs, labels, filename='tsne.png'):
  assert two_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = two_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.savefig(filename)


def read_analogies():
  """Reads through the analogy question file.
  Returns:
    questions: a [n, 4] numpy array containing the analogy question's
               word ids.
    questions_skipped: questions skipped due to unknown words.
  """
  questions = []
  questions_skipped = 0
  with open("analogical_reasoning_dataset.txt", "r") as analogy_f:
    for line in analogy_f:
      if line.startswith(":"):  # Skip comments.
        continue
      words = line.strip().lower().split(" ")
      ids = [dictionary[w.strip()] for w in words if w.strip() in dictionary.keys()]
      if None in ids or len(ids) != 4:
            questions_skipped += 1
      else:
        questions.append(np.array(ids))
  print("Eval analogy file: analogical_reasoning_dataset.txt")
  print("Questions: ", len(questions))
  print("Skipped: ", questions_skipped)
  return np.array(questions, dtype=np.int32)


def predict(analogy_a, analogy_b, analogy_c):
  # Normalized word embeddings of shape [vocab_size, emb_dim].
  #nemb = tf.nn.l2_normalize(embeddings, 1)
  nemb = final_embeddings
  #parameters = np.loadtxt("text8_parameters.txt", dtype='f', delimiter=' ')
  #nemb = parameters[:, :embedding_size]

  # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
  # They all have the shape [N, emb_dim]
  a_emb = nemb[analogy_a, :]  # a's embs
  b_emb = nemb[analogy_b, :]  # b's embs
  c_emb = nemb[analogy_c, :]  # c's embs

  # We expect that d's embedding vectors on the unit hyper-sphere is
  # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
  target = c_emb + (b_emb - a_emb)

  # Compute cosine distance between each pair of target and vocab.
  # dist has shape [N, vocab_size].
  dist = np.dot(target, np.transpose(nemb))

  # For each question (row in dist), find the top 4 words.
  return np.sort(dist)[::-1][:, :4]


def eval_analogy_questions():
  """Evaluate analogy questions and reports accuracy."""

  # How many questions we get right at precision@1.
  correct = 0

  try:
    total = analogy_questions.shape[0]
  except AttributeError as e:
    raise AttributeError("Need to read analogy questions.")

  start = 0
  while start < total:
    limit = start + 2500
    sub = analogy_questions[start:limit, :]
    idx = predict(sub[:, 0], sub[:, 1], sub[:, 2])
    start = limit
    for question in xrange(sub.shape[0]):
      for j in xrange(4):
        if idx[question, j] == sub[question, 3]:
          # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
          correct += 1
          break
        elif idx[question, j] in sub[question, :3]:
          # We need to skip words already in the question.
          continue
        else:
          # The correct label is not the precision@1
          break
  print()
  print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                            correct * 100.0 / total))
  return correct, total  

# Step 1: Set the hyper-parameters and the tests datasets
vocabulary_size = 50000

data_index = 0

batch_size = 128
embedding_size = 100         # Dimension of the embedding vector.
skip_window = 4              # How many words to consider left and right.
num_skips = 2*skip_window    # How many times to reuse an input to generate a label.
bag_window = skip_window     # How many words to consider left and right.
num_sampled = 5*batch_size   # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16              # Random set of words to evaluate similarity on.
valid_window = 100           # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# num_steps = 10001 # batch_size = 128 and skip_window = 8 ==> 8 wt / batch ==> 800.008 wt

test1 = 'test1.zip'
test2 = 'test2.zip'
test3 = 'test5.zip'
test4 = 'test6.zip'
test5 = 'test9.zip'
test6 = 'test10.zip'

filename = [test1, test2, test3, test4, test5, test6]

for test in range(1, 7):
  # Step 2: Read the data into a list of strings.
  corpus = read_data(filename[test-1])
  num_steps = len(corpus) * 2 * skip_window // batch_size
  print('test%d' % test, ' - Data size %d' % len(corpus), ' - num_steps %d' % num_steps)

  # Step 3: Build the dictionary and replace rare words with UNK token.
  data, count, dictionary, reverse_dictionary = build_dataset(corpus,
                                                            vocabulary_size)
  del corpus  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

  # Step 4: Build and train a skip-gram model.
  graph = tf.Graph()

  with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
      # Look up embeddings for inputs.
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)

      # Construct the variables for the NCE loss
      weights = tf.Variable(
          tf.truncated_normal([vocabulary_size, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size)))
      biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Alternative option: use sampled_softmax_loss instead of nce_loss
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=weights, biases=biases, labels=train_labels,
                       inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    # Alternative option: use GradientDescentOptimizer instead of AdagradOptimizer
    optimizer = tf.train.AdagradOptimizer(.5).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

  with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    time0 = time()
    init.run()
    print('Initialized')

    average_loss = 0
    log_loss = []
    for step in range(num_steps):
      batch_inputs, batch_labels = generate_batch_SG(
          batch_size, num_skips, skip_window)
      feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
      average_loss += loss_val

      if step % (num_steps//20) == 0:
        if step > 0:
          average_loss /= (num_steps//20)
        # The average loss is an estimate of the loss over the last 2000 batches.
        delta_t = time() - time0
        print('Average loss at step %d: %f. Delta t in seconds: %d' % (step, average_loss, delta_t))
        log_loss.append('Average loss at step %d: %f. Delta t in seconds: %d' % (step, average_loss, delta_t))
        average_loss = 0

#       # Note that this is expensive (~20% slowdown if computed every 500 steps)
#       if step % (num_steps/5) == 0 and step != 0: #if step % 25000 == 0:
#         sim = similarity.eval()
#         for i in xrange(valid_size):
#           valid_word = reverse_dictionary[valid_examples[i]]
#           top_k = 8  # number of nearest neighbors
#           nearest = (-sim[i, :]).argsort()[1:top_k + 1]
#           log_str = 'Nearest to %s:' % valid_word
#           for k in xrange(top_k):
#             close_word = reverse_dictionary[nearest[k]]
#             log_str = '%s %s,' % (log_str, close_word)
#           print(log_str)
    final_embeddings = normalized_embeddings.eval()
    embeddings = embeddings.eval()
    weights = weights.eval()
    biases = biases.eval()

#  # Step 5: Visualize the embeddings.
#  try:
#    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#    plot_only = 500
#    two_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#    plot_with_labels(two_dim_embs, labels, filename="test{}_SG.png".format(test))
#
#  except ImportError:
#    print('Please install sklearn, matplotlib, and scipy to show embeddings.')

  # Step 6: Analogical Reasoning task.
  analogy_questions = read_analogies()
  correct, total = eval_analogy_questions()

  # Step 7: Save the embeddings.
  hyperparameters = 'alpha=.5, vocabulary_size={}, batch_size={}, embedding_size={}, '.format(vocabulary_size, batch_size, embedding_size)
  hyperparameters = hyperparameters + 'skip_window={}, num_sampled={}, num_steps={}'.format(skip_window, num_sampled, num_steps)
  accuracy = "accuracy=%.1f%% (%d/%d), " % (correct * 100.0 / total, correct, total)
  header = "test{}_SG - ".format(test) + accuracy + hyperparameters

  parameters = np.c_[final_embeddings, embeddings, weights, biases]
  with open('pickles/2parameters_test{}_SG.txt'.format(test), 'wb') as fp:
    pickle.dump([header, parameters], fp)
  with open('pickles/2log_loss_test{}_SG.txt'.format(test), 'wb') as fp:
    pickle.dump([header, log_loss], fp)
    
  # Step 8: Build and train a CBOW model.
  graph = tf.Graph()

  with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size, bag_window * 2])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
      # Look up embeddings for inputs.
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embeds = tf.nn.embedding_lookup(embeddings, train_inputs)

      # Construct the variables for the NCE loss
      weights = tf.Variable(
          tf.truncated_normal([vocabulary_size, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size)))
      biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Alternative option: use sampled_softmax_loss instead of nce_loss
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=weights, biases=biases, labels=train_labels,
                       inputs=tf.reduce_sum(embeds, 1), num_sampled=num_sampled, num_classes=vocabulary_size))
    
    # Construct the SGD optimizer using a learning rate of 1.0.
    # Alternative option: use GradientDescentOptimizer instead of AdagradOptimizer
    optimizer = tf.train.AdagradOptimizer(.5).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

  with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    time0 = time()
    init.run()
    print('Initialized')

    average_loss = 0
    log_loss = []
    for step in range(num_steps):
      batch_inputs, batch_labels = generate_batch_CBOW(
          batch_size, bag_window)
      feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
      average_loss += loss_val

      if step % (num_steps//20) == 0:
        if step > 0:
          average_loss /= (num_steps//20)
        # The average loss is an estimate of the loss over the last 2000 batches.
        delta_t = time() - time0
        print('Average loss at step %d: %f. Delta t in seconds: %d' % (step, average_loss, delta_t))
        log_loss.append('Average loss at step %d: %f. Delta t in seconds: %d' % (step, average_loss, delta_t))
        average_loss = 0

#       # Note that this is expensive (~20% slowdown if computed every 500 steps)
#       if step % (num_steps/5) == 0 and step != 0: #if step % 25000 == 0:
#         sim = similarity.eval()
#         for i in xrange(valid_size):
#           valid_word = reverse_dictionary[valid_examples[i]]
#           top_k = 8  # number of nearest neighbors
#           nearest = (-sim[i, :]).argsort()[1:top_k + 1]
#           log_str = 'Nearest to %s:' % valid_word
#           for k in xrange(top_k):
#             close_word = reverse_dictionary[nearest[k]]
#             log_str = '%s %s,' % (log_str, close_word)
#           print(log_str)
    final_embeddings = normalized_embeddings.eval()
    embeddings = embeddings.eval()
    weights = weights.eval()
    biases = biases.eval()  

#  # Step 5: Visualize the embeddings.
#  try:
#    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#    plot_only = 500
#    two_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#    plot_with_labels(two_dim_embs, labels, filename="test{}_CBOW.png".format(test))
#
#  except ImportError:
#    print('Please install sklearn, matplotlib, and scipy to show embeddings.')

  # Step 6: Analogical Reasoning task.
  analogy_questions = read_analogies()
  correct, total = eval_analogy_questions()

  # Step 7: Save the embeddings.
  hyperparameters = 'alpha=.5, vocabulary_size={}, batch_size={}, embedding_size={}, '.format(vocabulary_size, batch_size, embedding_size)
  hyperparameters = hyperparameters + 'bag_window={}, num_sampled={}, num_steps={}'.format(bag_window, num_sampled, num_steps)
  accuracy = "accuracy=%.1f%% (%d/%d), " % (correct * 100.0 / total, correct, total)
  header = "test{}_CBOW - ".format(test) + accuracy + hyperparameters

  parameters = np.c_[final_embeddings, embeddings, weights, biases]
  with open('pickles/2parameters_test{}_CBOW.txt'.format(test), 'wb') as fp:
    pickle.dump([header, parameters], fp)
  with open('pickles/2log_loss_test{}_CBOW.txt'.format(test), 'wb') as fp:
    pickle.dump([header, log_loss], fp)
