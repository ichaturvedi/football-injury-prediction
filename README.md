# Football Injury Prediction using Trajectory Forecasting and Hydraulic Correction

This repository contains MATLAB and Simulink code for analysing football player trajectories, forecasting future player movement using Social-LSTM, applying a physics-inspired hydraulic correction model, and generating player-level injury risk indicators.

The framework combines:

- Multi-player tracking using ByteTrack
- Social-LSTM trajectory forecasting
- Team-aware hydraulic trajectory correction
- Trajectory-derived risk feature extraction
- Injury-associated player ranking
- Optional sensor-fusion using a football injury dataset

The code was developed as part of research into trajectory-based football injury risk analysis and temporal modelling of multi-agent sports systems.

[![Watch Demo](demo.mp4)
``
---

# Overview

The complete workflow is shown below:

```text
Football Video
      │
      ▼
ByteTrack
(Multi-Player Tracking)
      │
      ▼
Player Trajectories
      │
      ▼
Social-LSTM
(Trajectory Forecasting)
      │
      ▼
Hydraulic Response Model
(Simulink)
      │
      ▼
Corrected Trajectories
      │
      ▼
Speed / Agility / Stress
      │
      ▼
Risk Indicators
      │
      ▼
Player Ranking
```

---

# Repository Contents

```text
football-injury-prediction/

├── bytetracker/
├── applyPlayerParameterSimulinkResponse.m
├── assignStableTeamColoursFromVideo.m
├── buildSocialSequences.m
├── evaluateTrajectoryAccuracy.m
├── example_player_parameter_simulink_run.m
├── football_social_lstm_multiple_csv.mat
├── hydlib.slx
├── makeOccupancyGrid.m
├── preparePlayerPositions.m
├── runFootballByteSocialSimulinkParameters.m
├── sldemo_hydcyl4.slx
├── sldemo_hydcyl4.slxc
└── sldemo_hydcyl_data.mat
```

---

# Important Notes

This repository does **not** contain:

- Raw football videos
- Trained Social-LSTM model weights
- Full experimental tracking datasets
- Kaggle injury prediction dataset files

Users must provide these independently.

---

# External Dependencies

## 1. ByteTrack

The `bytetracker/` folder contains example tracking outputs and demonstrates the expected format used by the MATLAB pipeline.

To use this repository with a new dataset:

1. Run ByteTrack on your football videos.
2. Export player trajectories.
3. Save the generated tracking files in the same format as the provided examples.

Expected information includes:

- Frame number
- Track ID
- Bounding box coordinates
- Player position

The hydraulic framework operates on tracking outputs and does not perform object detection.

---

## 2. Social-LSTM Model

The forecasting stage requires a trained Social-LSTM model.

Users should:

- Train a Social-LSTM model on their own football trajectory dataset, or
- Use an existing trained Social-LSTM model

The repository assumes Social-LSTM predictions have already been generated before hydraulic correction is applied.

The file:

```text
football_social_lstm_multiple_csv.mat
```

provides an example of the data structure expected by the pipeline.

---

## 3. Injury Dataset for Sensor Fusion

An optional sensor-fusion stage can be implemented using the publicly available University Football Injury Prediction Dataset.

Dataset:

https://www.kaggle.com/datasets/yuanchunhong/university-football-injury-prediction-dataset

The dataset contains football player records including:

- Physical characteristics
- Football-specific metrics
- Physical fitness assessments
- Lifestyle factors
- Injury labels

In our experiments, a subset of features corresponding to video-derived indicators such as speed, agility and reaction-related measures can be used within a k-nearest-neighbour injury scoring component.

---

# MATLAB Scripts

## preparePlayerPositions.m

Processes tracked player coordinates and prepares trajectories for forecasting and hydraulic correction.

---

## makeOccupancyGrid.m

Builds Social-LSTM occupancy grids representing nearby player interactions.

---

## buildSocialSequences.m

Constructs Social-LSTM observation and prediction sequences.

---

## assignStableTeamColoursFromVideo.m

Estimates stable team assignments using jersey colour information extracted from video frames.

---

## applyPlayerParameterSimulinkResponse.m

Maps trajectory-derived player indicators to hydraulic response parameters and applies the Simulink-generated correction.

---

## runFootballByteSocialSimulinkParameters.m

Main pipeline script.

This script integrates:

- ByteTrack trajectories
- Social-LSTM predictions
- Team identification
- Hydraulic correction

and generates corrected forecasts for downstream analysis.

---

## evaluateTrajectoryAccuracy.m

Evaluates forecasting quality using:

- Average Displacement Error (ADE)
- Final Displacement Error (FDE)

Both raw and hydraulically corrected trajectories can be compared.

---

## example_player_parameter_simulink_run.m

Minimal demonstration of the hydraulic response generation process.

---

# Simulink Models

## hydlib.slx

Hydraulic subsystem library.

---

## sldemo_hydcyl4.slx

Hydraulic cylinder simulation used to generate temporal response signals.

---

## sldemo_hydcyl_data.mat

Supporting simulation parameters and example input data.

---

# Methodology

## Multi-Player Tracking

Player trajectories are obtained using ByteTrack.

Every player track contains:

- Unique track identifier
- Image coordinates
- Bounding boxes
- Temporal information

---

## Social-LSTM Forecasting

Social-LSTM predicts future player positions using:

- Historical movement
- Nearby player interactions
- Occupancy-grid representations

The forecast serves as the baseline trajectory.

---

## Hydraulic Correction

The baseline forecast is treated as a proposed motion sequence.

Trajectory-derived indicators such as:

- Speed
- Agility
- Stress

are mapped into hydraulic model parameters.

The resulting Simulink response produces a temporal correction that modifies the forecast trajectory.

---

# Evaluation Metrics

## Average Displacement Error (ADE)

Mean Euclidean distance between predicted and ground-truth positions across future prediction steps.

---

## Final Displacement Error (FDE)

Euclidean distance between the final predicted position and the final observed position.

---

