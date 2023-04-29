# Kalman-Filters

## Summary
In this Repository, I have implemented EKF and UKF localization algorithms using landmarks and their accuracy has been compared.

-----------------
## Dataset
Victoria Park Dataset: http://www-personal.acfr.usyd.edu.au/nebot/victoria_park.htm

-----------------

## EKF

### Demo

[![Watch the video](EKF/demo.png)](https://user-images.githubusercontent.com/20353960/235326934-a1f5302e-c073-4358-b96f-475755adb956.mp4)

### Accuracy

**average x error:** 0.039 \
**average y error:** 0.052 \
**average theta error:** 0.029


## UKF

### Demo

[![Watch the video](UKF/demo.png)](https://user-images.githubusercontent.com/20353960/235326935-b4da737c-c51b-415d-b6e6-bbb8c84d3d4d.mp4)

### Accuracy
**average x error:** 0.026 \
**average y error:** 0.023 \
**average theta error:** 0.030

-----------------

## Comparisons

| Metric      | EKF         | UKF |
| ----------- | ----------- | ----------- |
| average x error      | 0.039 | **0.026**|
| average y error      | 0.051 | **0.023**|
| average theta error  | **0.029** | 0.030|
