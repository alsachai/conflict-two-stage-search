# Two-Stage Collision Scenario Testing on CARLA + Autoware

This folder ports our two-stage collision scenario-testing approach to
**CARLA + Autoware** as an additional system-under-test, alongside the
Apollo + LGSVL implementation in the parent repository. It is built on top
of [DriveFuzz](https://github.com/dk-kling/drivefuzz), which we use only as
the CARLA-Autoware execution harness; the search method (conflict detection
and two-stage mutation) is our own.


## Environment Setup:

  - Environment installation and preparation follow DriveFuzz. See the
    DriveFuzz repository and its ```INSTALL.md```: build the CARLA 0.9.10.1
    PythonAPI egg and the ```carla-autoware``` Docker image as instructed
    there.
  - In each shell session, run DriveFuzz's ```init.sh``` and source the ROS
    environment before launching, otherwise Autoware bring-up will hang.
  - Replace DriveFuzz's ```run_carla.sh``` with the one in ```ga_carla``` to additionally create recording directory.
  - Start CARLA (listening on ```localhost:2000```) with ```run_carla.sh```.


## Testing with our method:

  - Build the Autoware container once: ```cd ../carla-autoware && ./build.sh```
  - Find an Autoware-drivable ego route:
    ```python egoSearch.py```
  - Generate NPC spawn points close to that route: ```python npcSearch.py```
  - Run the testing: ```python drive_experiment.py```
  - Hyperparameters are defined at the top of
    ```drive_experiment.py```


## Outputs:

  - Found collisions are saved to ```AccidentScenario/```, named
    ```accident-<collision type>-gen<G>-<timestamp>```.
  - CARLA recordings are saved to ```recordings/```; collision recordings
    are renamed ```recording_<collision type>_<timestamp>.log```.


## Replay:

  - Replay a recording headlessly (renders to frames and an mp4):
    ```python replay.py --mode capture```
  - Replay the recording for a found accident (the recording is located
    automatically from the accident's timestamp):
    ```python replay.py --mode capture AccidentScenario/accident-<...>```
