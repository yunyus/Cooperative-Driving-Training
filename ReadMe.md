Cooperative Driving Simulation

This project simulates an cooperative driving environment where the agents (city, vehicle) learns to navigate a road with pedestrians using Q-learning. The environment parameters, Q-learning implementation, and simulation scripts are modularized for clarity and maintainability.

## Project Structure

```
cooperative_driving_sim/
│
├── data/
│   ├── q_table_speed.npy
│   ├── q_table_transition.npy
│   ├── simulation_data/
│       ├── oneq_test_entering_probability_0.005.xlsx
│       ├── oneq_test_entering_probability_0.01.xlsx
│       ├── ...
├── src/
│   ├── config.py
│   ├── environment.py
│   ├── q_learning.py
│   ├── test_simulation_oneq.py
│   ├── test_simulation_twoq.py
│   ├── test_simulation_threeq.py
│   ├── train_simulation.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Ensure the `q_table_speed.npy` and `q_table_transition.npy` files are present in the `data` directory. If not, you can train the model.

## Usage

### Training

To train the Q-tables, run:

```
python src/train_simulation.py
```

### Testing

To run the simulation and generate results for `oneq` which means one-d environment, run:

```
python src/test_simulation_oneq.py
```

To run the simulation and generate results for `twoq` which means two-d environment, run:

```
python src/test_simulation_twoq.py
```

To run the simulation and generate results for `threeq` which means three-d environment, ensure you have CARLA installed and set up, then run:

```
python src/test_simulation_threeq.py
```

## CARLA Setup

For running `threeq`, you need to have CARLA installed. Follow the instructions on the [CARLA website](https://carla.org/) to set up CARLA on your system.
