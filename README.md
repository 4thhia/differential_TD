# differential TD (dTD)

This is the official repository for differential TD, a temporal-difference method for stochastic continuous dynamics.

## Setup

1. Build the Docker image:

   ```bash
   # Replace the CUDA version in docker/Dockerfile to match your environment
   # Replace "your_username" in build.sh with your username
   bash build.sh
   ```

2. Launch the container:

   ```bash
   # Replace "your_username" in launch.sh with your username
   bash launch.sh
   ```

3. Run the setup inside the container:

   ```bash
   # Inside the container
   # Replace "your_username" in setup.sh with your username
   bash setup.sh
   ```

## Usage

Run experiments from the `dtd/` directory:

- Training:
  ```bash
  bash runs/main.sh
  ```
- Hyperparameter tuning:
  ```bash
  bash runs/tune.sh
  ```
The most critical options are:
- `ENV_NAME`: Specifies the environment to run (e.g., hopper, ant).
- `TD`: Chooses the TD method to use (baseline, naive, or dtd).
- `NOISE_LVL`: Sets the noise level in the environment dynamics.

You can also adjust `MAX_BUDGET` and `MIN_BUDGET` depending on your available computational resources.


## Acknowledgements

This implementation is partially based on the following repositories:

- [PureJaxRL](https://github.com/luchris429/purejaxrl)
- [How to AutoRL](https://github.com/facebookresearch/how-to-autorl)
