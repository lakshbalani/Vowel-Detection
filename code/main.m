clc;
clear all;
close all;

[y,fs]=audioread('Source_Dont_ask_me_to_carry_an_oily_rag_like_that.wav');
N = length(y);

figure()
plot(y);
title('Original Signal');
ylabel('Amplitude');
xlabel('Time');

%% Implementation of Paper 1
% lowpass
fc = 2500;
[b,a] = butter(6,fc/(fs/2),'low');
y1 = filter(b,a,y);

figure()
plot(y1);
title('Lowpass Filtered Signal');
ylabel('Amplitude');
xlabel('Time');

%hilbert

y2 = hilbert(y1);

y3 = y1 + 1j * y2;

HE = abs(y3);

figure()
plot(HE);
title('Hilbert Envelope');

%apply single pole

HE1 = filter(1,[1 -1],HE);

figure()
plot(HE1);
title('Single Pole');

L=80;

slopes = zeros(size(HE1));
for i=L:size(HE1) - L -1
    for k=1:L
        slopes(i) = slopes(i) + abs(HE1(i + k + 1) - HE1(i + k - L));
    end
end
slopes = slopes / L;

slopes = slopes / max(slopes);

figure()
plot(slopes);
title('AMD');

% sigmoidal


th = 0.8*mean(slopes);
beta = 5*mean(slopes);

sig = 1./(1+exp(-beta*(slopes-th)));

figure()
plot(sig);
title('NL-AMD');

L = 0.1*fs;
Fogd1=diff(gausswin(L,6/L));


y5=conv(sig,Fogd1,'same');

y5=y5./max(y5);
envelope1 = y5;
y5(y5<0)=0;
y5(y5<0.01)=0;

y5=smooth(y5,320);

th = 0.008 * max(y5);
VOPs1 = zeros(size(y5));
for i=2:size(y5)-1
    if (y5(i) > th && y5(i) > y5(i-1) && y5(i) > y5(i+1))
        VOPs1(i) = 1;
    end
end
VOPs1(VOPs1 == 0) = NaN;


%% Enhancing the VOP and removing close peaks and ploting it with speech signal
figure()
subplot(211)
plot(y5)
title('VOP evidence plot');
subplot(212)
plot(y-mean(y))
hold on
stem(VOPs1,'*')
title('VOP in speech signal')


%% Implementation of Paper 2
p = 4000/18;
x(1,:)=bandpass(y(:,1),[15 p],fs);

for i=2:18
    x(i,:)=bandpass(y(:,1),[(i-1)*p i*p],fs);
end

%% half wave rectifier
a = zeros(size(x));
for i=1:18
    for j=1:N
        if x(i,j)>0
            a(i,j)=x(i,j);
        end
    end
end

%% low pass filter
for i=1:18
    lp(i,:)=lowpass(a(i,:),28,fs,'Steepness',0.95);
    % normalize
    lp(i,:)=lp(i,:)./max(lp(i,:));
end

%% fft win
for i=1:18
    temp=lp(i,:);
    temp=buffer(temp,20,19);

    temp=temp.*(hamming(20)*ones(1,length(temp)));

    temp=fft(temp,40);

    temp = sum(abs(temp(4:16,:)));

    temp1(i,:)=temp;

end

temp1=sum(temp1);

temp1=resample(temp1,80,fs);
temp1=resample(temp1,fs,80);
temp1=temp1/max(temp1);

figure()
plot(temp1);
title('Modulation Spectrum');
xlabel('Time');
ylabel('Amplitude');

temp1=filtfilt(hamming(1600),1,temp1);
y6=diff(temp1);
y7=buffer(y6,160,159);

y8=sum(y7);

y9=y8;

y9(y8<0)=0;

figure()
plot(y9);
title('Enhanced Modulation Spectrum');
xlabel('Time');
ylabel('Amplitude');


Fogd2=diff(gausswin(800));

y10=filter(Fogd2,1,y9);


%% Enhancing the VOP and removing close peaks and ploting it with speech signal

y10=y10./max(y10);
envelope2 = y10;
y10(y10<0)=0;
y10(y10<0.1)=0;

y10=smooth(y10,320);
th = 0.1;
VOPs2 = zeros(size(y10));
for i=2:size(y10)-1
    if (y10(i) > th && y10(i) > y10(i-1) && y10(i) > y10(i+1))
        VOPs2(i) = 1;
    end
end
VOPs2(VOPs2 == 0) = NaN;
figure()
subplot(211)
plot(y10)
title('VOP evidence plot');
subplot(212)
plot(y-mean(y))
hold on
stem(VOPs2,'*', Color = 'red')
title('VOP in speech signal')


%% Comparision
figure()
plot(envelope1)
hold on
plot(envelope2)
% legends
legend('Using AMD','Using Modulation Spectrum')
title('Vowel evidence plot for both methods')


figure()
plot(y-mean(y))
hold on
stem(VOPs1,'*', Color = 'green')
stem(VOPs2,'*', Color = 'red')
% legends
legend('Original Signal','Using AMD','Using Modulation Spectrum')
title('VOPs from both methods')
