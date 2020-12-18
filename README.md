# Vehicle State Estimation on a Roadway
This project implements the Error-State Extended Kalman Filter (ES-EKF) to localize a vehicle using data from the CARLA simulator. 

The starter code is provided by Coursera and can be found here.

## Preliminaries

### Vehicle State Initialization

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

## Prediction

### Motion Model

The motion model of the vehicle is given by the following equations

| Description | Equation                                                     |
| ----------- | ------------------------------------------------------------ |
| Position    | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cp%7D_k%20%3D%20%5Cboldsymbol%7B%5Cp%7D_%7Bk-1%7D%20%2B%20%7B%5CDelta%7Dt%5Cboldsymbol%7B%5Cv%7D_%7Bk-1%7D%20%2B%20%5Cfrac%7B%7B%5CDelta%7Dt%5E2%7D%7B2%7D(%5Cboldsymbol%7B%5CC%7D_%7Bns%7D%5Cboldsymbol%7B%5Cf%7D_%7Bk-1%7D%20%2B%20%5Cboldsymbol%7B%5Cg%7D)%0A"> |
| Velocity    | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cv%7D_%7Bk%7D%20%3D%20%5Cboldsymbol%7B%5Cv%7D_%7Bk-1%7D%20%2B%20%7B%5CDelta%7Dt(%5Cboldsymbol%7B%5CC%7D_%7Bns%7D%5Cboldsymbol%7B%5Cf%7D_%7Bk-1%7D%20%2B%20%5Cboldsymbol%7B%5Cg%7D)%0A"> |
| Orientation | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cq%7D_%7Bk%7D%3D%5Cboldsymbol%7B%5Cq%7D_%7Bk-1%7D%5Cotimes%5Cboldsymbol%7B%5Cq%7D(%5Cboldsymbol%7B%5Comega%7D_%7Bk-1%7D%7B%5CDelta%7Dt)%3D%5Cboldsymbol%7B%5COmega%7D(%5Cboldsymbol%7B%5Cq%7D(%5Cboldsymbol%7B%5Comega%7D_%7Bk-1%7D%7B%5CDelta%7Dt))%5Cboldsymbol%7B%5Cq%7D_%7Bk-1%7D%0A"> |

