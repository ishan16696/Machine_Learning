function idx1 = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx1 = zeros(size(X,1), 1);
%idx2 = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X,1)
for i=1:m
    M = repmat(X(i,:),K,1);
    M=(M-centroids);
    
    Z=M(:,1).^2 + M(:,2).^2;
       
    [val ind]=min(Z);
    idx1(i)=ind;
end




% 2nd method 
% m = size(X,1)
% for i = 1:m
%     distance_array = zeros(1,K);
%     for j = 1:K
%         distance_array(1,j) = sqrt(sum( power   (    (X(i,:)-  centroids(j,:))   ,2  )  )  );
%         
%     end
%     [d, d_idx] = min(distance_array);
%     idx2(i,1) = d_idx;
% end
% 
% idx1==idx2  % it is used for the checking 
% =============================================================

end

