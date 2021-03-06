function [hmm] = hmm_learn(letter, prev_letter, pixels, alpha)
% Learns parameters for an HMM from a given data table.
%
% Usage:
%
%    hmm = hmm_learn(LETTER, PREV_LETTER, PIXELS)
%
% Learns parameters for HMM from the specified dataset. If M is the number of 
% characters in the dataset, then LETTER is a M x 1 array of letter labels,
% PIXELS is a M x 64 binary matrix of pixel activations for each letter,
% and PREV_LETTER is a M x 1 array indicating the PREVIOUS correct letter
% at each position. If the position corresponds to the start of a word,
% PREV_LETTER should be -1.
%
% The output is a struct with the following fields:
%
%  hmm.pstart : 26 x 1 vector of P(X_1 = i)
%  hmm.ptrans : 26 x 26 matrix, where 
%               hmm.ptrans(i,j) is P(X_t = j | X_{t-1} = i)
%  hmm.pletters : 26 x 1 vector of P(X_t = i) (used only for Naive Bayes)
%  hmm.pobs : 26 x 64 matrix of P(O^j = 1 | X_t = i)

% Estimate letter transition probabilities:
pstart = zeros(26,1);
pletters = zeros(26,1);
ptrans = zeros(26,26);
pobs = zeros(26, 64);

% YOUR CODE GOES HERE
%%% pstart
start_word = find(prev_letter==-1);
for k = 1:26
    pstart(k) = (length(find(letter(start_word)==k))+alpha)/(length(start_word)+alpha);
end
 
%%% ptrans
for i = 1:26
    for j = 1:26
        ptrans(i,j) = (length(find(prev_letter==i&letter==j))+alpha)/(length(find(prev_letter==i))+alpha);
    end
end
 
%%% pobs
for k = 1:26
    for l = 1:64
        pobs(k,l) = (length(find(letter==k&pixels(:,l)==1))+alpha)/(length(find(letter==k))+alpha);
    end
end
 
%%% pletters
for k = 1:26
    pletters(k) = (length(find(letter==k))+alpha)/(length(letter)+alpha);
end

hmm.pstart = pstart;
hmm.ptrans = ptrans;
hmm.pobs = pobs;
hmm.pletters = pletters;

