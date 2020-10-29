import os
from mujoco_py import utils, load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import numpy as np
from scipy.spatial.transform import Rotation as R
import pdb
import sys
import matplotlib.pyplot as plt

def simulate_initialState(initial_state):
	H = 10
	TOLERANCE = 1e-3

	sim_state = sim.get_state()

	sim_state.qpos[:3] = initial_state[0:3, 0]
	sim_state.qpos[3:] = initial_state[3:7, 0]

	sim_state.qvel[:3] = initial_state[7:10,0]
	sim_state.qvel[3:] = initial_state[10:,0]

	sim.set_state(sim_state)
	sim.forward()

	data_arr = np.empty((0,13))

	while(True):
		sim.set_state(sim_state)
		sim.forward()
		for i in range(MAX_STEPS):
			blank_arr = np.zeros((1,13))

			blank_arr[0, :7] = sim.get_state().qpos
			blank_arr[0, 7:] = sim.get_state().qvel
			data_arr = np.append(data_arr, blank_arr, axis = 0)
			sim.step()
			# viewer.render()

			if i > 10:
				diff_data = data_arr[(i-H):i,:] - data_arr[(i-H-1):i-1,:]
				if np.max(np.linalg.norm(diff_data,axis=1)) < TOLERANCE:
					break
					
		if SAVE_SIMDATA:
			np.savetxt(SAVE_PATH, data_arr.T, delimiter=',')
			return data_arr
			break

def getPerturbedQuat(initial_quat, range_min, range_max):
	theta_min = range_min/BLOCK_D
	theta_max = range_max/BLOCK_D
	if(theta_min < -6.28):
		theta_min = -6.28
	if(theta_max > 6.28):
		theta_max = 6.28

	theta = np.random.uniform(theta_min, theta_max, 1)
	axis_ = np.random.uniform(range_min, range_max, (3,))
	axis_ = axis_/np.linalg.norm(axis_)
	perturb_quat = np.array([axis_[0]*np.sin(theta/2), axis_[1]*np.sin(theta/2)\
				, axis_[2]*np.sin(theta/2), np.cos(theta/2)])
	initial_quat_ = np.array([initial_quat[1], initial_quat[2], initial_quat[3],\
					initial_quat[0]])

	initial_rot = R.from_quat(initial_quat_)
	perturb_rot = R.from_quat(perturb_quat[:,0])
	res_rot_quat = (perturb_rot*initial_rot).as_quat()
	res_rot_quat = np.array([res_rot_quat[-1], res_rot_quat[0],\
					 res_rot_quat[1], res_rot_quat[2]])
	return res_rot_quat

MODEL_XML = "cube_toss.xml"
mj_path, _ = utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', MODEL_XML)
model = load_model_from_path(xml_path)

sim = MjSim(model)
# viewer = MjViewer(sim)

SAVE_SIMDATA = True  #flag to denote if simulated trajectories is to be saved
MAX_STEPS = 500		 #limit the trajectory length to 500 timesteps if the block does not come to rest by then
NUM_TOSSES = 11000   #num of trjaectories to be generated
BLOCK_D = 0.1        #The body length of the cube block
range_min = -0.1    #upper bound of the distribution
range_max = 0.1      #lower bound of the distribution
STIFFNESS = 2500     #stiffness setting currently used (update the settings in xml file)

#the reference initial state around which new initial states will be generated
initial_state = np.array([ 0.18629883,  0.02622872,  0.12283257, -0.52503014,  0.39360754,
       -0.29753734, -0.67794127,  0.01438053,  1.29095332, -0.21252927,
        1.46313532, -4.85439428,  9.86961928]).reshape(-1,1)

#generate trajectories
for i in range(NUM_TOSSES):
	perturbed_state_ = np.ones((13,1))
	perturbed_state_[:3,0] = initial_state[:3,0] + np.random.uniform(range_min, range_max,(3,))
	perturbed_state_[7:10,0] = initial_state[7:10,0] + np.random.uniform(range_min, range_max,(3,))
	perturbed_state_[3:7,0] = getPerturbedQuat(initial_state[3:7,0], \
								range_min, range_max)
	perturbed_state_[10:,0] = initial_state[10:,0] + np.random.uniform(range_min, range_max,(3,))
	print(str(i+1), " done")
	SAVE_PATH = f"../contactlearning/data/{STIFFNESS}/mujoco_sim"+str(i+1)+".csv"
	simulate_initialState(perturbed_state_)
