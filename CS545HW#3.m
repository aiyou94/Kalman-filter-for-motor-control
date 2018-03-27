>> clear
%Part a
% Filter order, cutoff, sampling, and nyquist frequency
>> fs=100;
>> nyq=fs/2;
>> fc=5;
>> [b,a]=butter(2,fc/nyq);
>> freqz(b,a);

%Part b
%Sample length
N=1000;
%Load noisy data
>> load noisy.data
>> yn=noisy(:,1);
>> xn=noisy(:,2);
>> un=noisy(:,3);
%Time vector and sample size vector
>> t=0:0.01:9.99;
>> n=1:N;
t=t';
n=n';
%Filter coeffecients
a =[1.0000   -1.5610    0.6414];
b =[0.0201    0.0402    0.0201];
%Run the filter
>> xfb=filter(b,a,yn);
%plot of the filtered & original data
figure(1)
>> plot(n,xfb,'r');
hold on
>> plot(n,xn,'k');
legend('Butter filtered','true');
>> title('filtered & original data');
>> xlabel('iterations')
%butter filter delay
>> db=finddelay(xn,xfb);
% here delay equals to five steps (0.05s)


%Part c
%Define the system.
a=0.5;  %a=1 for a constant, |a|<1 for a first order system.
b=3.5;
c=eye(1);

%Define the noise covariances.
Q=0.01;
R=1;

%Preallocate memory
load noisy.data
x=zeros(1,N);
xapriori=zeros(1,N);
xf=zeros(1,N);
residual=zeros(1,N);
papriori=ones(1,N);
paposteriori=ones(1,N);
k=zeros(1,N);

%Model uncertainty and measurement noise.
>> w=gaussmf((1:N),[sqrt(Q) 0]);
>> v=gaussmf((1:N),[sqrt(R) 0]);

%Initial condition on the state, x.
x_0=0.5;

%Initialization for state and a posteriori covariance.
xf_0=0.1;
paposteriori_0=1;


%Compute the first estimates of the state and the output
x(1)=a*x_0+b*un(1)+w(1);
yn(1)=c*x(1)+v(1);

%Predictor equations
xapriori(1)=a*xf_0;
residual(1)=yn(1)-c*xapriori(1);
papriori(1)=a*a*paposteriori_0+Q;
%Corrector equations
k(1)=c*papriori(1)/(c*c*papriori(1)+R);
paposteriori(1)=papriori(1)*(1-c*k(1));
xf(1)=xapriori(1)+k(1)*residual(1);

%Compute the rest of the values
for j=2:N,
    %Calculate the estimated state (based on model dynamics and previous state)
    x(j)=a*x(j-1)+b*un(j-1)+w(j);
    %Predictor or time-upadate equations
    xapriori(j)=a*xf(j-1);
    residual(j)=yn(j)-c*xapriori(j);
    papriori(j)=a*a*paposteriori(j-1)+Q;
    %Corrector equations (measurement update)
    k(j)=c*papriori(j)/(c*c*papriori(j)+R);
    paposteriori(j)=papriori(j)*(1-c*k(j));
    xf(j)=xapriori(j)+k(j)*residual(j);
end

j=1:N;
>> figure(2);
>> subplot(3,1,1);
>> plot(j,xn);
>> title('true');
>> subplot(3,1,2);
>> plot(j,x);
>> title('Kalman filtered');
>> subplot(3,1,3);
>> plot(j,x,'r');
>> title('true & filered data');
hold on
>> plot(j,xn,'k');
legend('filtered','true');
%plot posterior covariance
figure(3)
>> plot(j,paposteriori);
>> title('Posterior covariance vs iterations');
%plot gain
figure(4)
>> plot(j,k);
>> title('Gain vs iterations');
%delay for Kalman filter
>> dk=finddelay(xn,x);           % 1 in this case (0.01s)

