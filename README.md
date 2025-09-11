# From Conflicts to Collisions: A Two-Stage Collision Scenario-Testing Approach for Autonomous Driving Systems


## Requirements:

  - Apollo7.0
  - lgsvl 2021.3
  - apollo_map: see maps/apollo_map, copy maps/apollo_map/SanFrancisco to /apollo/modules/map/data/SanFrancisco
  - lgsvl_map: SanFrancisco_correct link:https://wise.svlsimulator.com/maps/profile/12da60a7-2fc9-474d-a62a-5cc08cb97fe8


## LGSVL Config:

  - Add map SanFrancisco_correct from Store to Library. Map link:https://wise.svlsimulator.com/maps/profile/12da60a7-2fc9-474d-a62a-5cc08cb97fe8
  - Add an API Only simulation.

## Testing with our method:

  - Run ```python main_ours.py``` to start the testing.
  - Define the scenario in ```config_ds_1.yaml``` and ```ds_1.json```