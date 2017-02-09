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
%%% pixel based probabilities
for i = 1:T
    for k = 1:N
        pobs(k,i)=exp(sum(log(hmm.pobs(k, logical(pix(i,:)))))+sum(log(1-hmm.pobs(k, ~logical(pix(i,:))))));
    end
end
 
%%% forward probabilities
% initialization
f = zeros(N, T);
f(:,1)=hmm.pstart.*pobs(:,1);
 
 
% iteration
for i = 2:T
    for k = 1:N
        f(k,i)=pobs(k,i)*sum(f(:,i-1).*hmm.ptrans(:,k));
    end
end
 
%%% backward probabilities
% initialization
b = zeros(N, T);
b(:,T)=ones(N,1);
 
% iteration
for i = T-1:-1:1
    for k = 1:N
        b(k,i)=sum(hmm.ptrans(k,:)'.*pobs(:,i+1).*b(:,i+1));
    end
end
 
%%% joint distribution
for i = 1:T
    for k = 1:N
        px(k,i) = f(k,i)*b(k,i);
    end
end
 

