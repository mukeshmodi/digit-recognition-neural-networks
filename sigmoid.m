function [ mat_s ] = sigmoid(x )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
mat_s=zeros(1,length(x));

for k=1:length(x)
mat_s(1,k)=1/(1+exp(-x(k)));

end
end

