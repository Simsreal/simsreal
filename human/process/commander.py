import json
import time

import zmq


def commander_proc(runtime_engine):
    cfg = runtime_engine.get_metadata("config")
    robot_pub_cfg = cfg["robot"]["pub"]
    pub = zmq.Context().socket(zmq.PUB)
    pub.bind(
        f"{robot_pub_cfg['protocol']}://{robot_pub_cfg['ip']}:{robot_pub_cfg['port']}"
    )

    while True:
        torques = runtime_engine.get_shm("torques").clone().squeeze(0)
        cmds = {
            "torques": torques.tolist(),
        }
        pub.send_string(json.dumps(cmds))
        time.sleep(1 / cfg["running_frequency"])
