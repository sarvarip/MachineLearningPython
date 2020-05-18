from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################

  alpha[:,0] = np.multiply(pi[:], B[:,O[0]])
  for t in np.arange(N-1):
      term = np.matmul(A.T, alpha[:,t])
      alpha[:,t+1] = np.multiply(B[:,O[t+1]], term)

  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################

  beta[:,N-1] = 1
  for t in np.arange(start=N-1, stop=0, step=-1):
      term = np.multiply(B[:,O[t]], beta[:,t])
      beta[:,t-1] = np.matmul(A, term)

  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward messages

  Inputs:
  - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})

  Returns:
  - prob: A float number of P(X_{1:N}=O)
  """
  N = alpha.shape[1]
  prob = np.sum(alpha[:,N-1])
  ###################################################
  # Q3.2 Edit here
  ###################################################

  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward messages

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(X_{1:N}=O)
  """
  alp1 = np.multiply(pi[:], B[:,O[0]])
  prob = np.sum(np.multiply(beta[:,0], alp1))
  ###################################################
  # Q3.2 Edit here
  ###################################################

  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path (in terms of the state index)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  ###################################################
  S = len(pi)
  N = len(O)
  delta = np.zeros([S, N])
  idx_vec = np.zeros([S, N])
  delta[:,0] = np.multiply(pi[:], B[:,O[0]])
  for t in np.arange(N-1):
      term = np.max(np.multiply(A, delta[:,t].reshape(-1, 1)), axis=0)
      idx = np.argmax(np.multiply(A, delta[:,t].reshape(-1, 1)), axis=0)
      idx = idx.ravel()
      idx_vec[:,t] = idx
      term = term.reshape(-1, 1)
      delta[:,t+1] = np.multiply(B[:,O[t+1]].reshape(-1,1), term).ravel()

  Zt = np.argmax(delta[:,t])
  path.append(Zt)
  for t in np.arange(start=N-1, stop=0, step=-1):
      Zt = idx_vec[Zt,t-1].astype(int)
      path.append(Zt)
  path.reverse()

  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  #model_file = sys.argv[1]
  #Osymbols = sys.argv[2]

  model_file = "hmm_model.json"
  Osymbols = "AGCGTA"

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()
