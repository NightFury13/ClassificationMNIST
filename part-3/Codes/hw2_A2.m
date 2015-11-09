clear all;
[data labels] = loadDigits(2000, 'train'); % This data had dimensions 784x2000
D = tangent_d(data, data, 1);
options.dims = 1:20;
[Y, R, E] = Isomap(D, 'k', 7, options);
ind0 = find(labels==0);
ind1 = find(labels==1);
ind2 = find(labels==2);
ind3 = find(labels==3);
ind4 = find(labels==4);
ind5 = find(labels==5);
ind6 = find(labels==6);
ind7 = find(labels==7);
ind8 = find(labels==8);
ind9 = find(labels==9);
A = Y.coords(2);
B = A{1};

[m n] = size(ind1);
for i =1:n
    D1(:,i) = B(:,ind1(i));
end
[m n] = size(ind2);
for i =1:n
    D2(:,i) = B(:,ind2(i));
end
[m n] = size(ind3);
for i =1:n
    D3(:,i) = B(:,ind3(i));
end
[m n] = size(ind4);
for i =1:n
    D4(:,i) = B(:,ind4(i));
end
[m n] = size(ind5);
for i =1:n
    D5(:,i) = B(:,ind5(i));
end
[m n] = size(ind6);
for i =1:n
    D6(:,i) = B(:,ind6(i));
end
[m n] = size(ind7);
for i =1:n
    D7(:,i) = B(:,ind7(i));
end
[m n] = size(ind8);
for i =1:n
    D8(:,i) = B(:,ind8(i));
end
[m n] = size(ind9);
for i =1:n
    D9(:,i) = B(:,ind9(i));
end
[m n] = size(ind0);
for i =1:n
    D0(:,i) = B(:,ind0(i));
end
figure
plot(D1(1,:),D1(2,:),'r.');
hold;
plot(D7(1,:),D7(2,:),'b.');
title('Clusters of 1 and 7 using Tangent Distance with 1 in red and 7 in blue');
figure
plot(D4(1,:),D4(2,:),'r.');
hold;
plot(D9(1,:),D9(2,:),'b.');
title('Clusters of 4 and 9 using Tangent Distance with 4 in red and 9 in blue');


figure
plot(D1(1,:),D1(2,:),'r.');
hold all
plot(D2(1,:),D2(2,:),'b.');

plot(D3(1,:),D3(2,:),'g.');

plot(D4(1,:),D4(2,:),'y.');

plot(D5(1,:),D5(2,:),'m.');

plot(D6(1,:),D6(2,:),'k.');

plot(D7(1,:),D7(2,:),'c.');

plot(D8(1,:),D8(2,:),'ro');

plot(D9(1,:),D9(2,:),'bo');

plot(D0(1,:),D0(2,:),'go');
title('Clusters for all the digits using Tangent Distance');