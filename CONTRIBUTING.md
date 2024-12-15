# Contributing to Simsreal

## Important notes
* Please make sure you run `pre-commit install` to install pre-commit hooks at **each** submodule you work on.
<!-- * We are relying on Isaac Sim to incorporate physics constraints in simulations, so there is no need to model physical constraints on our own. -->

## World (or USDs)
Welcome to create or modify existing worlds. Feel free to create a PR in [environment](https://github.com/Simsreal/environment) to share your world in `isaac_sim_env/usds/`.

## Sensors
We are actively adding below sensors to human `Grace`:
* `contact_sensor`
* `audio_sensor`
* others not included in the list but human can sense and available in Isaac Sim now or future.

If you want to add a new sensor or enhance/modify existing sensors, please import `grace.usd` in [here](https://github.com/Simsreal/environment/tree/main/isaac_sim_env/usds), and create a copy with added sensors with the corresponding [OmniGraph/Action Graph](https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/sensor_simulation_physics_sensors.html).

## TODOs (not in near future)
* Github workflows for pre-commit
* Branch protection
