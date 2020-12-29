# Vehicle State Estimation on a Roadway
This project implements the Error-State **Extended Kalman Filter** (ES-EKF) to localize a vehicle using data from the [CARLA](https://carla.org/) simulator. The following diagram shows a graphical representation of the system.

<img src="images\diagram.png" style="zoom: 67%;" />



This project is the final programming assignment of the [State Estimation and Localization for Self-Driving Cars](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars?) course from [Coursera](https://www.coursera.org/). The starter code is provided by the [University of Toronto](https://www.utoronto.ca/).

The **Kalman Filter** algorithm updates a state estimate through two stages:

1. *prediction* using the motion model
2. *correction* using the measurement model

[TOC]

## 1. Preliminaries

### 1.1. Vehicle State Initialization

The **vehicle state** at each time step consists of position, velocity, and orientation (parameterized by a unit quaternion); the inputs to the **motion model** are the IMU specific force and angular rate measurements.

| Description      | Equation |
| ---------------- | -------- |
| Vehicle State    | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cx%7D_k%3D%5B%5Cboldsymbol%7B%5Cp%7D_k%2C%20%5Cboldsymbol%7B%5Cv%7D_k%2C%20%5Cboldsymbol%7B%5Cq%7D_k%5D%5E%7BT%7D%20%5Cin%20R%5E10%0A"> |
| IMU Measurements | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cu%7D_k%3D%5B%5Cboldsymbol%7B%5Cf%7D_k%2C%20%5Cboldsymbol%7B%5Comega%7D_k%5D%5E%7BT%7D%20%5Cin%20R%5E6%0A"> |

This section of code initializes the variables for the ES-EKF solver.

```python
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i  = 0
lidar_i = 0
```

## 2. Prediction

The main filter loop operates by first **predicting** the next state (vehicle pose and velocity). The predicted vehicle state integrates the high-rate IMU measurements by using a nonlinear motion model.

### 2.1. Motion Model

The motion model of the vehicle is given by the following set of equations

| Description | Equation                                                     |
| ----------- | ------------------------------------------------------------ |
| Position    | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cp%7D_k%20%3D%20%5Cboldsymbol%7B%5Cp%7D_%7Bk-1%7D%20%2B%20%7B%5CDelta%7Dt%5Cboldsymbol%7B%5Cv%7D_%7Bk-1%7D%20%2B%20%5Cfrac%7B%7B%5CDelta%7Dt%5E2%7D%7B2%7D(%5Cboldsymbol%7B%5CC%7D_%7Bns%7D%5Cboldsymbol%7B%5Cf%7D_%7Bk-1%7D%20%2B%20%5Cboldsymbol%7B%5Cg%7D)%0A"> |
| Velocity    | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cv%7D_%7Bk%7D%20%3D%20%5Cboldsymbol%7B%5Cv%7D_%7Bk-1%7D%20%2B%20%7B%5CDelta%7Dt(%5Cboldsymbol%7B%5CC%7D_%7Bns%7D%5Cboldsymbol%7B%5Cf%7D_%7Bk-1%7D%20%2B%20%5Cboldsymbol%7B%5Cg%7D)%0A"> |
| Orientation | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cq%7D_%7Bk%7D%3D%5Cboldsymbol%7B%5Cq%7D_%7Bk-1%7D%5Cotimes%5Cboldsymbol%7B%5Cq%7D(%5Cboldsymbol%7B%5Comega%7D_%7Bk-1%7D%7B%5CDelta%7Dt)%3D%5Cboldsymbol%7B%5COmega%7D(%5Cboldsymbol%7B%5Cq%7D(%5Cboldsymbol%7B%5Comega%7D_%7Bk-1%7D%7B%5CDelta%7Dt))%5Cboldsymbol%7B%5Cq%7D_%7Bk-1%7D%0A"> |

### 2.2. Predicted State

The **predicted** vehicle state is therefore given by the equations

| Description             | Equation                                                     | Variable  |
| ----------------------- | ------------------------------------------------------------ | --------- |
| *Predicted* State       | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Ccheck%7B%5Cx%7D%7D_%7Bk%7D%3D%5B%5Cboldsymbol%7B%5Ccheck%7B%5Cp%7D%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5Ccheck%7B%5Cv%7D%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5Ccheck%7B%5Cq%7D%7D_%7Bk%7D%5D%5ET%0A"> | `x_check` |
| *Predicted* Position    | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Ccheck%7B%5Cp%7D%7D_k%20%3D%20%5Cboldsymbol%7B%5Cp%7D_%7Bk-1%7D%20%2B%20%7B%5CDelta%7Dt%5Cboldsymbol%7B%5Cv%7D_%7Bk-1%7D%20%2B%20%5Cfrac%7B%7B%5CDelta%7Dt%5E2%7D%7B2%7D(%5Cboldsymbol%7B%5CC%7D_%7Bns%7D%5Cboldsymbol%7B%5Cf%7D_%7Bk-1%7D%20%2B%20%5Cboldsymbol%7B%5Cg%7D)%0A"> | `p_check` |
| *Predicted* Velocity    | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Ccheck%7B%5Cv%7D%7D_%7Bk%7D%20%3D%20%5Cboldsymbol%7B%5Cv%7D_%7Bk-1%7D%20%2B%20%7B%5CDelta%7Dt(%5Cboldsymbol%7B%5CC%7D_%7Bns%7D%5Cboldsymbol%7B%5Cf%7D_%7Bk-1%7D%20%2B%20%5Cboldsymbol%7B%5Cg%7D)%0A"> | `v_check` |
| *Predicted* Orientation | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Ccheck%7B%5Cq%7D%7D_%7Bk%7D%3D%5Cboldsymbol%7B%5Cq%7D_%7Bk-1%7D%5Cotimes%5Cboldsymbol%7B%5Cq%7D(%5Cboldsymbol%7B%5Comega%7D_%7Bk-1%7D%7B%5CDelta%7Dt)%0A"> | `q_check` |

This section of code iterates through the IMU inputs and updates the state.

```python
for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update state with IMU inputs
    q_prev = Quaternion(*q_est[k - 1, :]) # previous orientation as a quaternion object
    q_curr = Quaternion(axis_angle=(imu_w.data[k - 1]*delta_t)) # current IMU orientation
    c_ns = q_prev.to_mat() # previous orientation as a matrix
    f_ns = (c_ns @ imu_f.data[k - 1]) + g # calculate sum of forces
    p_check = p_est[k - 1, :] + delta_t*v_est[k - 1, :] + 0.5*(delta_t**2)*f_ns
    v_check = v_est[k - 1, :] + delta_t*f_ns
    q_check = q_prev.quat_mult_left(q_curr)
```

### 2.3. Error State Linearization

In the previous section, we propagated the nominal state (`x_check`) forward in time. However, this prediction ignores noise and perturbations. The ES-EKF algorithm captures these deviations in the error state vector. 

The next step is to consider the linearized error dynamics of the system.

| Description       | Equation                                                     |
| ----------------- | ------------------------------------------------------------ |
| Error State       | <img src="https://render.githubusercontent.com/render/math?math=%5Cdelta%5Cboldsymbol%7B%5Cx%7D_%7Bk%7D%3D%5B%5Cdelta%5Cboldsymbol%7B%5Cp%7D_%7Bk%7D%2C%20%5Cdelta%5Cboldsymbol%7B%5Cv%7D_%7Bk%7D%2C%20%5Cdelta%5Cboldsymbol%7B%5Cphi%7D_%7Bk%7D%5D%5ET%20%5Cin%20R%5E9%0A"> |
| Error Dynamics    | <img src="https://render.githubusercontent.com/render/math?math=%5Cdelta%5Cboldsymbol%7B%5Cx%7D_%7Bk%7D%3D%5Cboldsymbol%7B%5CF%7D_%7Bk-1%7D%5Cdelta%5Cboldsymbol%7B%5Cx%7D_%7Bk-1%7D%2B%5Cboldsymbol%7B%5CL%7D_%7Bk-1%7D%5Cboldsymbol%7B%5Cn%7D_%7Bk-1%7D%0A"> |
| Measurement Noise | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cn%7D_%7Bk%7D%5Csim%20N(%5Cboldsymbol%7B0%7D%2C%20%5Cboldsymbol%7BQ%7D_%7Bk%7D)%0A"> |

The Jacobians and the noise covariance matrix are defined as follows

| Description                 | Equation                                                     | Variable |
| --------------------------- | ------------------------------------------------------------ | -------- |
| Motion Model Jacobian       | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5CF%7D_%7Bk-1%7D%3D%5Cbegin%7Bbmatrix%7D%5Cboldsymbol%7B%5CI%7D%26%5Cboldsymbol%7B%5CI%7D%5Ccdot%5CDelta%20t%260%5C%5C0%26%5Cboldsymbol%7B%5CI%7D%26-%5B%5Cboldsymbol%7B%5CC%7D_%7Bns%7D%5Cboldsymbol%7B%5Cf%7D_%7Bk-1%7D%5D_%7B%5Ctimes%7D%5CDelta%20t%5C%5C0%260%26%5Cboldsymbol%7B%5CI%7D%5Cend%7Bbmatrix%7D%0A"> | `f_jac`  |
| Motion Model Noise Jacobian | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5CL%7D_%7Bk-1%7D%3D%5Cbegin%7Bbmatrix%7D0%260%5C%5C%5Cboldsymbol%7B%5CI%7D%260%5C%5C0%26%5Cboldsymbol%7B%5CI%7D%5Cend%7Bbmatrix%7D%0A"> | `l_jac`  |
| IMU Noise Covariance        | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5CQ%7D_%7Bk%7D%3D%5CDelta%20t%5E2%5Cbegin%7Bbmatrix%7D%5Cboldsymbol%7BI%7D%5Ccdot%5Csigma_%7Bacc%7D%5E2%260%5C%5C0%26%5Cboldsymbol%7BI%7D%5Ccdot%5Csigma_%7Bgyro%7D%5E2%5Cend%7Bbmatrix%7D%0A"> | `q_cov`  |

where ***I*** is the 3 by 3 identity matrix.

This section of code calculates the motion model Jacobian

```python
    # 1.1 Linearize the motion model and compute Jacobians
    f_jac = np.eye(9) # motion model jacobian with respect to last state
    f_jac[0:3, 3:6] = np.eye(3)*delta_t
    f_jac[3:6, 6:9] = -skew_symmetric(c_ns @ imu_f.data[k - 1])*delta_t
```

### 2.4. Propagate Uncertainty

In the previous section, we computed the Jacobian matrices of the motion model. We will use these matrices to propagate the state uncertainty forward in time.

The uncertainty in the state is captured by the state covariance (uncertainty) matrix. The uncertainty grows over time until a measurement arrives.

| Description                  | Equation                                                     | Variable      |
| ---------------------------- | ------------------------------------------------------------ | ------------- |
| *Predicted* State Covariance | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Ccheck%7B%5CP%7D%7D_%7Bk%7D%3D%5Cboldsymbol%7B%5CF%7D_%7Bk-1%7D%5Cboldsymbol%7B%5CP%7D_%7Bk-1%7D%5Cboldsymbol%7B%5CF%7D_%7Bk-1%7D%5E%7BT%7D%2B%5Cboldsymbol%7B%5CL%7D_%7Bk-1%7D%5Cboldsymbol%7B%5CQ%7D_%7Bk-1%7D%5Cboldsymbol%7B%5CL%7D_%7Bk-1%7D%5E%7BT%7D%0A"> | `p_cov_check` |

This section of code calculates the state uncertainty

```python
    # 2. Propagate uncertainty
    q_cov = np.zeros((6, 6)) # IMU noise covariance
    q_cov[0:3, 0:3] = delta_t**2 * np.eye(3)*var_imu_f
    q_cov[3:6, 3:6] = delta_t**2 * np.eye(3)*var_imu_w
    p_cov_check = f_jac @ p_cov[k - 1, :, :] @ f_jac.T + l_jac @ q_cov @ l_jac.T
```

## 3. Correction

### 3.1. Measurement Availability

The IMU data arrives at a faster rate than either GNSS or LIDAR sensor measurements.

The algorithm checks the measurement availability and calls a function to correct our prediction.

```python
    # 3. Check availability of GNSS and LIDAR measurements
    if imu_f.t[k] in gnss_t:
        gnss_i = gnss_t.index(imu_f.t[k])
        p_check, v_check, q_check, p_cov_check = \
            measurement_update(var_gnss, p_cov_check, gnss.data[gnss_i], p_check, v_check, q_check)
    
    if imu_f.t[k] in lidar_t:
        lidar_i = lidar_t.index(imu_f.t[k])
        p_check, v_check, q_check, p_cov_check = \
            measurement_update(var_lidar, p_cov_check, lidar.data[lidar_i], p_check, v_check, q_check)
```

### 3.2. Measurement Model

The measurement model is the same for both sensors. However, they have different noise covariance.

| Description             | Equation                                                     |
| ----------------------- | ------------------------------------------------------------ |
| Measurement Model       | <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0A%5Cboldsymbol%7By%7D_%7Bk%7D%20%26%3D%20%5Cboldsymbol%7Bh%7D(%5Cboldsymbol%7Bx%7D_%7Bk%7D)%2B%5Cboldsymbol%7Bv%7D_%7Bk%7D%20%5C%5C%0A%26%20%3D%5Cboldsymbol%7BH%7D_%7Bk%7D%5Cboldsymbol%7Bx%7D_%7Bk%7D%2B%5Cboldsymbol%7Bv%7D_%7Bk%7D%20%5C%5C%0A%26%20%3D%20%5Cboldsymbol%7Bp%7D_%7Bk%7D%2B%5Cboldsymbol%7Bv%7D_%7Bk%7D%0A%5Cend%7Bsplit%7D%0A"> |
| GNSS Measurement Noise  | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7Bv%7D_%7Bk%7D%20%5Csim%20N(0%2C%20%5Cboldsymbol%7BR%7D_%7B%5CGNSS%7D)%0A"> |
| LIDAR Measurement Noise | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7Bv%7D_%7Bk%7D%20%5Csim%20N(0%2C%20%5Cboldsymbol%7BR%7D_%7B%5CLIDAR%7D)%0A"> |

### 3.3. Measurement Update

Measurements are processed sequentially by the EKF as they arrive; in our case, both the GNSS receiver and the LIDAR provide position updates.

| Description                | Equation                                                     | Variable |
| -------------------------- | ------------------------------------------------------------ | -------- |
| Measurement Model Jacobian | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7BH%7D_%7Bk%7D%3D%5Cbegin%7Bbmatrix%7D%5Cboldsymbol%7BI%7D%260%260%5C%5C%5Cend%7Bbmatrix%7D%0A"> | `h_jac`  |
| Sensor Noise Covariance    | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7BR%7D%3D%5Cboldsymbol%7BI%7D%5Ccdot%5Csigma_%7B%5Csensor%7D%5E2%0A"> | `r_cov`  |
| Kalman Gain                | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7BK%7D_%7Bk%7D%3D%5Cboldsymbol%7B%5Ccheck%7BP%7D%7D_%7Bk%7D%5Cboldsymbol%7BH%7D_%7Bk%7D%5E%7BT%7D(%5Cboldsymbol%7BH%7D_%7Bk%7D%5Cboldsymbol%7B%5Ccheck%7BP%7D%7D_%7Bk%7D%5Cboldsymbol%7BH%7D_%7Bk%7D%5E%7BT%7D%2B%5Cboldsymbol%7BR%7D)%5E%7B-1%7D%0A"> | `k_gain` |

This section of code defines the measurement update function

```python
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain
    r_cov = np.eye(3)*sensor_var
    k_gain = p_cov_check @ h_jac.T @ np.linalg.inv((h_jac @ p_cov_check @ h_jac.T) + r_cov)
```

We use the Kalman gain to compute the error state. Considering the innovation or difference between our predicted vehicle position, `p_check`, and the measured position, `y_k`. 

| Description | Equation                                                     | Variable      |
| ----------- | ------------------------------------------------------------ | ------------- |
| Error State | <img src="https://render.githubusercontent.com/render/math?math=%5Cdelta%5Cboldsymbol%7Bx%7D_%7Bk%7D%3D%5Cboldsymbol%7BK%7D_%7Bk%7D(%5Cboldsymbol%7By%7D_%7Bk%7D-%5Cboldsymbol%7B%5Ccheck%7Bp%7D%7D_%7Bk%7D)%0A"> | `error_state` |

The error state is then used to update the nominal state vector. We also calculate the corrected nominal state covariance matrix, `p_cov_hat`.

| Description                  | Equation                                                     | Variable    |
| ---------------------------- | ------------------------------------------------------------ | ----------- |
| *Corrected* Position         | <img src="https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cboldsymbol%7Bp%7D%7D_%7Bk%7D%3D%5Ccheck%7B%5Cboldsymbol%7Bp%7D%7D_%7Bk%7D%2B%5Cdelta%5Cboldsymbol%7Bp%7D_%7Bk%7D%0A"> | `p_hat`     |
| *Corrected* Velocity         | <img src="https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cboldsymbol%7Bv%7D%7D_%7Bk%7D%3D%5Ccheck%7B%5Cboldsymbol%7Bv%7D%7D_%7Bk%7D%2B%5Cdelta%5Cboldsymbol%7Bv%7D_%7Bk%7D%0A"> | `v_hat`     |
| *Corrected* Orientation      | <img src="https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cboldsymbol%7Bq%7D%7D_%7Bk%7D%3D%5Cboldsymbol%7Bq%7D(%5Cdelta%5Cboldsymbol%7B%5Cphi%7D_%7Bk%7D)%5Cotimes%5Ccheck%7B%5Cboldsymbol%7Bq%7D%7D_%7Bk%7D%0A"> | `q_hat`     |
| *Corrected* State Covariance | <img src="https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cboldsymbol%7BP%7D%7D_%7Bk%7D%3D(%5Cboldsymbol%7BI%7D-%5Cboldsymbol%7BK%7D_%7Bk%7D%5Cboldsymbol%7BH%7D_%7Bk%7D)%5Ccheck%7B%5Cboldsymbol%7BP%7D%7D_%7Bk%7D%0A"> | `p_cov_hat` |

Finally, the function returns the corrected state and state covariants.

```python
    # 3.2 Compute error state
    error_state = k_gain @ (y_k - p_check)

    # 3.3 Correct predicted state
    p_hat = p_check + error_state[0:3]
    v_hat = v_check + error_state[3:6]
    q_hat = Quaternion(axis_angle=error_state[6:9]).quat_mult_left(Quaternion(*q_check))

    # 3.4 Compute corrected covariance
    p_cov_hat = (np.eye(9) - k_gain @ h_jac) @ p_cov_check

    return p_hat, v_hat, q_hat, p_cov_hat
```

## 4. Vehicle Trajectory

The algorithm generates visualizations by running the following command in the terminal.

```bash
python es_ekf.py
```

### 4.1. Ground Truth and Estimate

This figure shows a comparison between the trajectory estimate and the ground truth.

<img src="images\estimated-trajectory.gif" alt="estimated-trajectory"  />

### 4.2. Estimation Error and Uncertainty Bounds

This figure shows our estimator error relative to the ground truth. The dashed red lines represent three standard deviations from the ground truth, according to our estimator. These lines indicate how well our model fits the actual dynamics of the vehicle and how well the estimator is performing overall. The estimation error should remain within the three-sigma bounds at all times.

<img src="images\error-plots.png" alt="error-plots" />