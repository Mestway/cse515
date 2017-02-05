from __future__ import print_function
import os
import math
import sys

from matplotlib import pyplot

import random
import numpy as np

CompareToTrainTrue = False

log_history_list = []

# data: a list of list representing k training data, each with n features
# init_mus: the initialized value of mus, each mu is a n-dimensional array
# init_sigmas: the initailized values of sigmas for each cluster, each a 
# init_cp: the initialized component probablity
def em_algorithm(init_mus, init_sigmas, init_log_cps, feature_num, data):

  K = len(init_mus)
  N = len(data)

  mus = init_mus
  sigmas = init_sigmas
  log_cps = init_log_cps

  log_likelihood = log_gaussian_likelihood(mus, sigmas, log_cps, data)

  likihood_history = []

  iteration = 0
  while True:

    old_mus = mus
    old_sigmas = sigmas
    old_log_cps = log_cps
    old_log_likelihood = log_likelihood

    log_likelihood = log_gaussian_likelihood(mus, sigmas, log_cps, data)

    if (iteration > 2 and (log_likelihood - old_log_likelihood < 0.001)) or iteration > 100:
      break

    if (iteration != 0):
      likihood_history.append(log_likelihood)

    iteration += 1

    ## E-step
    #responsibility

    log_r = [] # 2-d array: first dimension: sample id, second dimension: component prob
    for i in range(0, N):
      log_r_i = []
      item = []
      for k in range(0, K):
        item.append(log_cps[k] + log_gaussian(mus[k], sigmas[k], data[i]))
      
      sum_item = log_sum_using_sum_log(item)

      for k in range(0, K):
        log_r_i.append( (item[k] - sum_item) )

      log_r.append(log_r_i)

    ## M-step

    # update each mu
    for k in range(0, K):
      item1 = 0
      item2 = 0
      for i in range(0, N):
        item1 += np.multiply(np.exp(log_r[i][k]), data[i])
        item2 += np.exp(log_r[i][k])
      mus[k] = np.multiply(item1, 1/item2)

    for k in range(0, K):
      # foreach dimension
      sigmas[k] = []
      for t in range(0, feature_num):
        head_item = 0
        tail_items = []
        for i in range(0, N):
          head_item += (data[i][t] - mus[k][t]) * (data[i][t] - mus[k][t]) * np.exp(log_r[i][k])
          tail_items.append(log_r[i][k])
        sigmas[k].append(np.sqrt(head_item / np.exp(log_sum_using_sum_log(tail_items))))

    # update each log_cps
    for k in range(0, K):
      log_items = []
      for i in range(0, N):
        log_items.append(log_r[i][k])
      log_cps[k] = log_sum_using_sum_log(log_items) - np.log(N)

  if CompareToTrainTrue:
    truth = []
    true_file = open("wine-true.train", 'r')
    for line in true_file:
      truth.append(int(line.split()[0]))

    cls_list = []
    # the prediction
    for i in range(0, N):
      log_max_prob = -999999;
      cls = -1
      for k in range(0, K):
        logp = log_cps[k] + log_gaussian(mus[k],sigmas[k],data[i])
        if logp > log_max_prob:
          log_max_prob = logp
          cls = k
      cls_list.append(cls)

    cls_map = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    score = 0
    for mp in cls_map:
      temp_score = 0
      for t in range(0, len(truth)):
        if (mp[cls_list[t]] == truth[t]):
          temp_score += 1
      if temp_score > score:
        score = temp_score
    # correctness of rating
    print(score)

  model = []
  model.append([K, feature_num])
  for k in range(0, K):
    cluster_feature = []
    cluster_feature.append(np.exp(log_cps[k]))
    cluster_feature.extend(mus[k])
    cluster_feature.extend(sigmas[k])
    model.append(cluster_feature)

  log_history_list.append(likihood_history)

  return model

def log_sum_using_sum_log(log_items):
  max_item = np.amax(log_items)
  return max_item + np.log(sum(np.exp(item_i - max_item) for item_i in log_items))

# mu is a vector [mu_1,...,mu_n]
# sigma is a list, representing a diagonal matrix
# x is also a list
# all are n dimensional
def log_gaussian(mu, sigma, x):
  n = len(x)
  y = np.subtract(x, mu)
  first_item = - 0.5 * np.dot(y, np.multiply([1. / s if s != 0 else 1 for s in sigma], y))
  second_item = - n / 2. * np.log(2. * np.pi) - 0.5 * np.log(np.dot(mu, mu))
  return first_item + second_item

def log_gaussian_likelihood(mus, sigmas, log_cps, data):
  K = len(mus)
  N = len(data)

  result = 0
  for i in range(0, N):
    log_items = []
    for k in range(0, K):
      log_items.append(log_cps[k] + log_gaussian(mus[k], sigmas[k], data[i]))
    result += log_sum_using_sum_log(log_items)

  return result

def main(num_cluster, data_file, model_file):

  data = []
  feature_num = 0
  input_file = open(data_file, 'r')
  
  first_line = True
  for line in input_file:
    if (first_line):
      feature_num = int(line.split()[1])
      first_line = False
    else:
      data.append([float(x) for x in line.split()])

  init_sigmas = []  
  init_log_cps = []
  for i in range(0, num_cluster):
    sigma = []
    for j in range(0, feature_num):
      sigma.append(1)
    init_sigmas.append(sigma)

    init_log_cps.append(-np.log(num_cluster))

  # method 1
  init_mus = []
  for i in range(0, num_cluster):
    init_mus.append(data[random.randint(0, len(data)-1)])
  
  # method 2
  #max_mins = []
  #for j in range(0, feature_num):
  #  max_mins.append([-99999,99999])
  #for j in range(0, feature_num):
  #  for i in range(0, len(data)):
  #    if (data[i][j] > max_mins[j][0]):
  #      max_mins[j][0] = data[i][j]
  #    if (data[i][j] < max_mins[j][1]):
  #      max_mins[j][1] = data[i][j]

  #init_mus = []
  #for i in range(0, num_cluster):
  #  mu = []
  #  for j in range(0, feature_num):
  #    mu.append(random.uniform(max_mins[j][0]-1, max_mins[j][1]+1))
  #  init_mus.append(mu)

  model = em_algorithm(init_mus, init_sigmas, init_log_cps, feature_num, data)

  output_file = open(model_file, 'w')
  #for param_list in model:
  #  print(' '.join(str(x) for x in param_list), file=output_file)

if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2], sys.argv[3])

  #for it in range(1,11):
    #main(it, sys.argv[2], sys.argv[3])

  lines = []
  for t in range(0, len(log_history_list)):
    line, = pyplot.plot(range(0,len(log_history_list[t])), log_history_list[t], label=str(t+1))
    lines.append(line)
  pyplot.legend(lines, [str(x) for x in range(1, len(lines) +1)])
  pyplot.show()
