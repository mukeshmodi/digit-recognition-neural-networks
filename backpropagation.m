clear all
close all
test_data = loadMNISTImages('t10k-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
traning_data = loadMNISTImages('train-images-idx3-ubyte');
labels2= loadMNISTLabels('t10k-labels-idx1-ubyte');

%Parameters
Input_Neurons=784;
Hidden_Neurons=100;
Output_Neurons=10;
alpha=0.15;
lambda=0.2;
% Converting the output matrix to a suitable form
O =zeros(length(labels),10);
%load('MNIST13.mat');
% Intitalizing weights 
Theta1=randn(Input_Neurons+1,Hidden_Neurons);
Theta2=randn(Hidden_Neurons+1,Output_Neurons);

% Converting the images to 2
for z=1:6
    z
for k=1:length(labels)
    if(labels(k)== 0)
        labels(k)=10;
    end 
end
for k=1:length(labels2)
    if(labels2(k)==0)
        labels2(k)=10;
    end 
end 
% Making the desired output file 
for k= 1:length(labels)
    O(k,labels(k))=1;
end


%forward Propagation
h=zeros(1,100);
for k=1:60000
I=traning_data(:,k);
I=[1;I];

a=I'*Theta1;
h=sigmoid(a);
h=[1,h];
l=h*Theta2;
y=sigmoid(l);


% now implement backward propagation 

del=(y-O(k,:)).*sigmoid(l).*(ones(1,10)-sigmoid(l));

delta_error=(del'*h)';
del_input=h.*(ones(1,101 )-h).*(Theta2*del')';

del_error_input=del_input'*I';
alpha=.5;
del_error_input=del_error_input(2:end,:);
Theta1=Theta1-alpha*del_error_input';
Theta2=Theta2-alpha*delta_error;



end
end

%Testing
y1=zeros(10000,10);

%forward Propagation
h=zeros(1,100);
for k=1:10000
T=test_data(:,k);
T=[1;T];

a=T'*Theta1;
h=sigmoid(a);
h=[1,h];
l=h*Theta2;
y=sigmoid(l);
y1(k,:)=y;

end

%Accuracy

count=1;

for k= 1:length(labels2)
O2(k,labels2(k))=1;
end

[r,c]=size(y1);

for i=1:r
    for j=1:c
        diff=1-y1(i,j);
        if(diff<.2)
            y1(i,j)=1;
        else
            y1(i,j)=0;
        end
    end
end

count1=1;

for i=1:r
    for j=1:c
        f=1;
        if((O2(i,j)==1)&&(y1(i,j)==1))
            count1=count1+1;
           
        end
           
        end
end
acc=count1/10000
