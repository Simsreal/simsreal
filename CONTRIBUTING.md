# Contributing to Simsreal

Thank you for your interest in contributing to Simsreal. Especially there is no existing reference to what AGI should be, the journey to AGI is challenging but exciting and fun.

We aim to be a pioneer who creates AGI, build use-cases around it that opens an entirely fresh horizon for general public and businesses.

## Table of Contents
- [Beliefs on AGI](#beliefs-on-agi)
- [Prerequisites](#prerequisites)
- [Principles](#principles)
- [What you can contribute](#what-you-can-contribute)
- [Intelligence](#intelligence)
- [Worlds](#world)
- [Sensors](#sensors)

## Beliefs on AGI
To create a shared AGI vision and stable progressions to Simsreal's success, and most importantly establishing core guidelines for you to familiarize with Simsreal and contribute to it, we encourage all Simsreal contributors have the same beliefs on the ways Simsreal approach to AGI:

* All modern AI, includings Large Language Models (LLMs) do not scale to or even reach AGI.
* Enriched sensory experience is crucial for AGI's consciousness.
* AGI relies on *on-the-fly* learning where it takes place at "now" and "here". Traditional training-deployment does not apply here.
* AGI's consciousness is emergent and a product of memory, perceptions, and available context.
* AGI's consciousness is driven/ guided by instincts, and thus their behaviours.
* AGI can only act on controllable joints/limbs.

## Prerequisites
Once completed these items, you are ready to contribute to Simsreal:

1. Follow [README.md](README.md) to setup the environment for development, that includes pip packages in `requirements-dev.txt`, ROS2 Humble, Isaac Sim, etc.
2. Able to start Simsreal application, e.g. by `python host.py --config isaac/grace`.
3. Use `host.py` as your entry point to understand and study how Simsreal works.
4. Run `pre-commit install` at **every** submodule you work on, which is critical for accessible code quality for everyone including yourself.

## Principles
Your contributions should have these principles in mind:
1. Modularity and reusability: the addition/removal of codes do not break the system and can be easily re-used and integrated.
2. Type-annotated functions, parameters, and classes.
3. Inheritance of abstract classes: most classes implemented in `human` have their inheritance from abstracted concepts defined in `intelligence` submodule. This is a necessary step to reinforce our [Beliefs on AGI](#beliefs-on-agi) and make sure your contributions are consistent with the overall design.


For example, most classes implemented in `human` have their inheritance from abstracted concepts defined in `intelligence` submodule. This is a necessary step to reinforce our [Beliefs on AGI](#beliefs-on-agi) and make sure your contributions are consistent with the overall design.


## What you can contribute
We believed the effectiveness of contribution is maximized when you are familiar and feel interested with specific components of Simsreal you work on, and we are happily offering you a general guide for you to kickstart your contribution.

1. [Intelligence](#intelligence)

  Related topics:
  * Test-time Learning
  * Human Cognitives (e.g. Perceptions, Instincts)
  * PyTorch Modules as Memory/ Experience Machine
  * Neuro-Symbolic AI (for planning)

2. [World](#world) / [Sensors](#sensors)

  Related topics:
  * USD/ 3D-content creations (3D objects)
  * Physics-based Simulation (Collisions, gravity, etc.)
  * ROS2, ROS2 Bridge (Robot-Python interface, Pub/Sub)

3. General workflows (Not in near future)

  Related topics:
  * Branch protection
  * Github workflows

## Intelligence
Developing intelligence in Simsreal is developing the [emergence of consciousness](https://github.com/Simsreal/human/blob/main/src/images/flow_draft_2.png), which involves components including `Constraint`, `Context`, `Instinct`, `Memory`, `Perceptors`, and others. All consciousness components are abstracted in `intelligence` submodule.

There is a sample workflow for you to follow, where we create `photoreceptor` as `Perceptor`:

1. Create a new file or modify an file in `human/perceptors/vision.py`
2. Develop a new `Perceptor` class, `Photoreceptor(Perceptor)`, and implement the expected method by `Perceptor` which is `perceive()`
```python
class Photoreceptor(Perceptor):
    def __init__(self):
        super().__init__("photoreceptor")

    def perceive(self, *args, **kwargs):
        ...
```

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

## World
Welcome to create or modify existing worlds. Feel free to create a PR in [environment](https://github.com/Simsreal/environment) to share your world in `isaac_sim_env/usds/`.

### General guidelines
* Make incremental changes to world complexity, e.g. adding more objects, more complex scenes, etc.
* The objects you created in the world are enabled with `physics` and interactable with [Sensors](#sensors).
* Often ask yourself: how does this change affect the human's consciousness upon perceiving it?

## Sensors
Building human with sensors is crucial to make `Perceptions` enriched and its subsequent processing.

Our goal is to make humans created have as much sensory experience as possible, and we are actively adding below sensors to human `Grace`:
* `contact_sensor`
* `audio_sensor`
* others not included in the list but human can sense and available in Isaac Sim now or future.

You can get started by importing [grace.usd](https://github.com/Simsreal/environment/tree/main/isaac_sim_env/usds) in Isaac Sim and work on a copy of it.

Remember to the corresponding [OmniGraph/Action Graph](https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/sensor_simulation_physics_sensors.html) for your created sensors to make sure they are published over `ROS2 Bridge`.
