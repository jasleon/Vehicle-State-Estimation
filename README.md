# Vehicle State Estimation on a Roadway
This project implements the Error-State Extended Kalman Filter (ES-EKF) to localize a vehicle using data from the CARLA simulator. 

The starter code is provided by Coursera and can be found here.

## Prediction

### Motion Model

The motion model of the vehicle is given by the following equations

| Description | Equation                                                     |
| ----------- | ------------------------------------------------------------ |
| Position    | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cp%7D_k%20%3D%20%5Cboldsymbol%7B%5Cp%7D_%7Bk-1%7D%20%2B%20%7B%5CDelta%7Dt%5Cboldsymbol%7B%5Cv%7D_%7Bk-1%7D%20%2B%20%5Cfrac%7B%7B%5CDelta%7Dt%5E2%7D%7B2%7D(%5Cboldsymbol%7B%5CC%7D_%7Bns%7D%5Cboldsymbol%7B%5Cf%7D_%7Bk-1%7D%20%2B%20%5Cboldsymbol%7B%5Cg%7D)%0A"> |
| Velocity    | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cv%7D_%7Bk%7D%20%3D%20%5Cboldsymbol%7B%5Cv%7D_%7Bk-1%7D%20%2B%20%7B%5CDelta%7Dt(%5Cboldsymbol%7B%5CC%7D_%7Bns%7D%5Cboldsymbol%7B%5Cf%7D_%7Bk-1%7D%20%2B%20%5Cboldsymbol%7B%5Cg%7D)%0A"> |
| Orientation | <img src="https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Cq%7D_%7Bk%7D%3D%5Cboldsymbol%7B%5Cq%7D_%7Bk-1%7D%5Cotimes%5Cboldsymbol%7B%5Cq%7D(%5Cboldsymbol%7B%5Comega%7D_%7Bk-1%7D%7B%5CDelta%7Dt)%3D%5Cboldsymbol%7B%5COmega%7D(%5Cboldsymbol%7B%5Cq%7D(%5Cboldsymbol%7B%5Comega%7D_%7Bk-1%7D%7B%5CDelta%7Dt))%5Cboldsymbol%7B%5Cq%7D_%7Bk-1%7D%0A"> |

