function [ px pobs ] = hmm_fb(hmm, pix)
% Forward Backward algorithm for HMM.
%
% Usage:
%
%   [ px pobs ] = hmm_fb(HMM, PIXELS)
%
% Runs forward-backward algorithm to compute marginal probabilities PX and
% given a model HMM and a input word PIXELS. The inputs and outputs are
% described as follows: if T is the number of letters in the word, 
%
%   PX : 26 x T matrix of marginal letter probabilities.
%   POBS : 26 x T matrix of letter evidence probabilities P( O_t | X_t )
%   (i.e., the component of the probability that depends only on the pixels
%   at a given position t.)
%
%   HMM : The HMM model, output by HMM_LEARN.
%   PIXELS : A T x 64 binary matrix of pixel activations.

% 26 states
N = 26;

% Pre-allocate the outputs 
pobs = zeros(N, size(pix, 1)); % Just the pixel based probabilities 
                               % (i.e., no inference required)
px = zeros(N, size(pix, 1));   % marginal probabilities from forward-backward

% YOUR CODE GOES HERE
 

