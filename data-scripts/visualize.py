import os
from mujoco_py import utils, load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import numpy as np
from scipy.spatial.transform import Rotation as R
import pdb
import sys
import matplotlib.pyplot as plt

def visualizer_rollout(data):
	while(True):
		for i in range(data.shape[1]):
			x_des = data[0:3, i]
			quat_des = data[3:7,i]

			dx_des = data[7:10,i]
			dw_des = data[10:,i]

			sim_state = sim.get_state()

			sim_state.qpos[:3] = x_des
			sim_state.qpos[3:] = quat_des

			sim_state.qvel[:3] = dx_des
			sim_state.qvel[3:] = dw_des

			sim.set_state(sim_state)
			sim.forward()
			viewer.render()


if (len(sys.argv) != 2):
	print("USAGE: python[3] visualize [SIM_CSV]")
	sys.exit()

MODEL_XML = "cube_toss.xml"
SIM_CSV = sys.argv[1]

sim_data_ = np.loadtxt(SIM_CSV, delimiter = ",")
if(sim_data_.shape[1] < sim_data_.shape[0]):
	sim_data_ = sim_data_.T

mj_path, _ = utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', MODEL_XML)
model =load_model_from_path(xml_path)

sim = MjSim(model)
viewer = MjViewer(sim)

visualizer_rollout(sim_data_)


