# Contributing to Simsreal

Thank you for your interest in contributing to Simsreal. Especially there is no existing reference to what AGI should be, the journey to AGI is challenging but exciting and fun.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Intelligence](#intelligence)
- [Worlds](#worlds)
- [Sensors](#sensors)

## Prerequisites
1. Follow [README.md](README.md) to setup the environment for development, that includes pip packages in `requirements-dev.txt`, ROS2 Humble, Isaac Sim, etc.
2. Able to start Simsreal application, e.g. by `python host.py --config isaac/grace`.
3. Run `pre-commit install` at **every** submodule you work on, which is critical for accessible code quality for everyone including yourself.

## Intelligence
Developing intelligence in Simsreal is developing the emergence of consciousness, which involves components including `Constraint`, `Context`, `Instinct`, `Memory`, `Perceptors`, and others.

To let any humans use these components and to follow a principle of code modularity, there is a sample workflow for you to follow. Here is an example where you want to create `photoreceptor` as `Perceptor`:

1. Create a new file or modify an file in `human/perceptors/vision.py`
2. Develop a new `Perceptor` class, `Photoreceptor(Perceptor)`, and implement the expected method by `Perceptor` which is `perceive()`
```python
class Photoreceptor(Perceptor):
    def __init__(self):
        super().__init__("photoreceptor")

    def perceive(self, *args, **kwargs):
        ...
```

All consciousness components are abstracted in `intelligence`, you are welcome to get familiar with them and how they work to contribute emergence of consciousness.

3. Add the new `Perceptor` class to `__all__` in `human/perceptors/__init__.py`
```python
from .vision import Photoreceptor, ...

__all__ = [
    "...",
    "Photoreceptor",
]
```

4. Update `self` attributes in `host.py` to include the new `Perceptor` class, that is to put `photoreceptor: Photoreceptor` in `Host.Perceptors`
```python
class Host:
    Perceptors: Dict[str, Perceptor] = {
        "...", ...
        "photoreceptor": Photoreceptor,
    }
```
5. Update `simulation_config/isaac/grace.yaml` to include the new `Perceptor` class and its configuration
```yaml
humans:
  - name: grace
    perceptors:
      - name: photoreceptor
        configuration: null
      - ...
```

6. Run the simulation with updated `simulation_config`
```bash
python host.py --config isaac/grace
```


## World (or USDs)
Welcome to create or modify existing worlds. Feel free to create a PR in [environment](https://github.com/Simsreal/environment) to share your world in `isaac_sim_env/usds/`.

## Sensors
We are actively adding below sensors to human `Grace`:
* `contact_sensor`
* `audio_sensor`
* others not included in the list but human can sense and available in Isaac Sim now or future.

If you want to add a new sensor or enhance/modify existing sensors, please import [grace.usd](https://github.com/Simsreal/environment/tree/main/isaac_sim_env/usds) in Isaac Sim, and create a copy with added sensors with the corresponding [OmniGraph/Action Graph](https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/sensor_simulation_physics_sensors.html).

## TODOs (not in near future)
* Github workflows for pre-commit
* Branch protection
