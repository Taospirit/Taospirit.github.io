---
layout: post
title: EKF_Code_Reading
subtitle: ekf代码学习笔记
date: 2019-06-26
author: lintao
header-img: img/post-bg-desk.jpg
catalog: true
tags:
  - Optimization
  - Control
---

## 1. Code show

## 2. Filter design

in this simulaton, the robot has a state vector includes 4 states at time $t$
$$ X_t = [x_t, y_t, \theta_t, v_t] $$
$x$, $y$ are a 2D x-y position, $\theta$ is orientation, and v isvelociyy.

in the code, "xEst" means the state vector.

And $P_t$ is covariace matrix of the state,

$Q$ is covariance matrix of process noise,

$R$ is covariance matrix of observation noise at time $t$

The robot has a speed sensor and a gyro sensor.

So, the input vector can be used as each time step

$$u_t = [v_t, w_t]$$

Also, the robot has a GNSS sensor, it means that the robot can observe x-y position at each time.

$$z_t = [x_t, y_t] $$

The input and observation vector includes sensor noise.
In the code, "observation" function generates the input and observation vector with noise.

## 3. Motion Model

The robot model is

$$ \dot{\phi}=w $$

So, the motion model is
$$x_{t+1}=Fx_t+Bu_t$$

where


$$F = \begin{bmatrix}
{1}&{0}&{0}&{0}\\
{0}&{1}&{0}&{0}\\
{0}&{0}&{1}&{0}\\
{0}&{0}&{0}&{1}\\
\end{bmatrix}$$

$$B = \begin{bmatrix}
{cos(\phi)dt}&{0}&\\
{sin(\phi)dt}&{0}&\\
{0}&{dt}&\\
{1}&{0}&\\
\end{bmatrix}$$

$d_t$ is a time interval.
This is implemented at below:

Its Jacobian matrix is 

$$J_F = \begin{bmatrix}
{\frac{dx}{dx}}&{\frac{dx}{dy}}&{\frac{dx}{d\phi}}&{\frac{dx}{d\phi}}&{\frac{dx}{dv}}\\
{\frac{dy}{dx}}&{\frac{dy}{dy}}&{\frac{dy}{d\phi}}&{\frac{dy}{d\phi}}&{\frac{dy}{dv}}\\
{\frac{d\phi}{dx}}&{\frac{d\phi}{dy}}&{\frac{d\phi}{d\phi}}&{\frac{d\phi}{d\phi}}&{\frac{d\phi}{dv}}\\
{\frac{dv}{dx}}&{\frac{dv}{dy}}&{\frac{dv}{d\phi}}&{\frac{dv}{d\phi}}&{\frac{dv}{dv}}\\
\end{bmatrix}$$

$$=\begin{bmatrix}
{1}&{0}&{-vsin(\phi)dt}&{cos(\phi)det}\\
{0}&{1}&{vcos(\phi)dt}&{sin(\phi)dt}\\
{0}&{0}&{1}&{0}\\
{0}&{0}&{0}&{1}\\
\end{bmatrix}$$

## 4.Observation Model
The robot can get x-y position information from GPS.
So GPS Observation model is 

$$z_t=Hx_t$$

where

$$B=\begin{bmatrix}
{1}&{0}&{0}&{0}\\
{0}&{1}&{0}&{0}\\
\end{bmatrix}$$

Its jacobian matrix is

$$J_H=\begin{bmatrix}
{1}&{0}&{0}&{0}\\
{0}&{1}&{0}&{0}\\
\end{bmatrix}
$$

## 5. Extented Kalman Filter

Localization porcess using Extended Kalman Filter: EKF is 

### 1. Predict

$$ x_{pred} = F_{x_t}+Bu_t $$

$$ P_{pred} = J_fP_tJ_f^{T} + Q$$

### 2. Update

$$ Z_{pred}= Hx_{pred}$$
$$ y = z - z_{pred} $$
$$ S = J_HP_{pred}J_H^{T}+R$$
$$ K = P_{pred}J_H^{T}S^{-1}$$
$$ x_{t+1} = x_{pred}+ Ky $$
$$ P_{t+1} = (I - KJ_H)P_{pred} $$
