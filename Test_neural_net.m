
%forward Propagation
h=zeros(1,100);
y1=zeros(10000,10);
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
