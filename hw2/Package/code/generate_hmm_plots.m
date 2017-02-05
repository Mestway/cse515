%% Load the data
load ../data/ocr_data.mat

% EXPLANATION OF THE DATA:
%
%  trainset.letter = the actual letters
%  trainset.prev_letter = the previous letter (-1 means start)
%  trainset.pixels = pixel values
%  trainset.wordidx{i} = row indices of the i'th word in the other fields
%
% You can plot the pixels of the i'th character in image form as follows:
% >> imagesc(reshape(trainset.pixels(i,:), 8, 8)'); axis image; colormap gray;

%% Train HMM and NB, plot the model
hmm = hmm_learn(trainset.letter, trainset.prev_letter, trainset.pixels);

% Plot transition matrix
figure('Name', 'Transition Probabilities');
letters = arrayfun(@(x){char(x)}, 97:97+25)
imagesc(hmm.ptrans); colormap gray;
set(gca,'XTickLabel', letters);
set(gca,'XTick', 1:26);
set(gca,'YTickLabel', letters);
set(gca,'YTick', 1:26);
xlabel('Next letter');
ylabel('Previous Letter');
title('Transition Probabilities');
print -djpeg -r72 plot_1.1.jpg;

% Plot observation probabilities
figure('Name', 'Observation Model');
p = reshape(hmm.pobs(1,:), 8, 8)'; 
imagesc(p); colormap gray; 
set(gca,'XTick',[]);
set(gca,'YTick',[]);
title('Observation model for letter ''a''');
print -djpeg -r72 plot_1.2.jpg;

%% Test HMM and NB
t = CTimeleft(numel(testset.wordidx));
errs = zeros(size(testset.wordidx));
errs_nb = zeros(size(testset.wordidx));
for i = 1:numel(testset.wordidx)
    t.timeleft();
    
    % Test HMM model
    idx = testset.wordidx{i};

    % Compute marginal probabilities of letters and take predictions
    [px,pobs] = hmm_fb(hmm, testset.pixels(idx, :));

    % Take maximum probability states at each point
    [pmax,yhat_hmm] = max(px);

    % Compute probability according to NB and take predictions
    px_nb = % YOUR CODE GOES HERE
            % Hint: Use pobs which was already computed for you by HMM_FB.
    
    [pmax_nb,yhat_nb] = max(px_nb);
    
    % Compute errors on this word
    errs(i) = sum(yhat_hmm ~= testset.letter(idx)');
    errs_nb(i) = sum(yhat_nb ~= testset.letter(idx)');
end

% Compute accuracies
n = numel(testset.letter);
fprintf('HMM Letter Accuracy: %g%%\n', (1-sum(errs)/n)*100);
fprintf('HMM Word Accuracy: %g%%\n', (1-mean(errs>0))*100);
fprintf('NB Letter Accuracy: %g%%\n', (1-sum(errs_nb)/n)*100);
fprintf('NB Word Accuracy: %g%%\n', (1-mean(errs_nb>0))*100);


