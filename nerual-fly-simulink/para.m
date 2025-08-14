clc;clear;close all;

a_hat_shape_0 = 4;

lambda_a = 0.01;
Q = eye(a_hat_shape_0) * 0.1;
R = eye(3) * 0.1;
R_inv = inv(R);
K = eye(3) * 50.0;
Lambda = eye(3) * 40.0;
g_vector = [0.0; 0.0; 9.81];

m = 1;

phi = [0.1 0.2 -0.3 1.0;
       0.1 0.2 -0.3 1.0;
       0.1 0.2 -0.3 1.0;];
