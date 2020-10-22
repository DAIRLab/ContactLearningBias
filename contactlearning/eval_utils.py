import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pdb
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline

# try:
#   %tensorflow_version 2.x
# except Exception:
#   pass

# %load_ext tensorboard

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch.nn as nn
import torch.nn.functional as F

from RNNPredictor import RNNPredictor, CubeTossDataset
import learning_utils 

import warnings
import argparse

def getTheta(pred_curr_quat, act_curr_quat):

	pred_curr_rot = R.from_quat(pred_curr_quat)
	act_curr_rot = R.from_quat(act_curr_quat)

	diff_rot = act_curr_rot*(pred_curr_rot.inv())
	axis_norm = np.linalg.norm(diff_rot.as_quat()[:3])

	return 2*np.arcsin(axis_norm)*(180/np.pi)

def getRotationalError(input_sequence, full_pred_state, state_labels = None, rolled_time = 100, tw = 8, single_step = False):
	"""
	This method computes the rotational error in degrees

	Input:
	input_sequence: numpy array of shape (tw,1,13) 
	full_pred_state: numpy array of shape (1,13)
	state_label: numpy array of shape (13) 

	Output:
	err_rot: numpy array of shape ()  which is a vector representing the average rotational error
	"""

	# pred_last_state = input_sequence[-1,0,:]

	# pred_last_quat = pred_last_state[3:7].numpy()
	# pred_last_quat = np.array([pred_last_quat[1], pred_last_quat[2], pred_last_quat[3], pred_last_quat[0]])/np.linalg.norm(pred_last_quat)
	
	pred_curr_quat = full_pred_state[0,3:7]
	pred_curr_quat = np.array([pred_curr_quat[1], pred_curr_quat[2], pred_curr_quat[3], pred_curr_quat[0]])/np.linalg.norm(pred_curr_quat)

	# pred_rot_angle = getRotationalAngle(pred_last_quat, pred_curr_quat)

	if not single_step:
		# act_last_state = state_labels[tw+rolled_time - 1,:].numpy()
		# act_last_quat = act_last_state[3:7]
		# act_last_quat = np.array([act_last_quat[1], act_last_quat[2], act_last_quat[3], act_last_quat[0]])/np.linalg.norm(act_last_quat)

		act_curr_state = state_labels[rolled_time,:].numpy()
		act_curr_quat = act_curr_state[3:7]
		act_curr_quat = np.array([act_curr_quat[1], act_curr_quat[2], act_curr_quat[3], act_curr_quat[0]])/np.linalg.norm(act_curr_quat)
	else:
		# act_last_state = pred_last_state
		# act_last_quat = pred_last_quat

		act_curr_state = state_labels[0].numpy()
		act_curr_quat = act_curr_state[3:7]
		act_curr_quat = np.array([act_curr_quat[1], act_curr_quat[2], act_curr_quat[3], act_curr_quat[0]])/np.linalg.norm(act_curr_quat)

	# act_rot_angle = getRotationalAngle(act_last_quat, act_curr_quat)

	return np.linalg.norm(full_pred_state[0,:3] - act_curr_state[:3]),\
		getTheta(act_curr_quat, pred_curr_quat)

def get_angular_velocity(quat_prev, quat_curr, dt = 0.0067):
	"""
	This method computes the angular velocity given a pair of quaternions representing orientation at two different time-steps.

	Inputs:
	quat_prev, quat_curr: numpy array of shape = [1,4] representing the two quaternions in the form (x,y,z,w)
	dt: the time-step value

	Output:
	angular_rate: numpy array of shape = [3,] giving angular velocity along the three axes 
	"""
	times = [0, dt]
	angles = np.concatenate((quat_prev, quat_curr), axis = 0)
	rotations = R.from_quat(angles)
	spline = RotationSpline(times, rotations)
	angular_rate = (spline(times, 1))

	return angular_rate[-1]

def compute_state_vel(initial_states, output, dt = 0.0067):
	"""
	This method calculates the velocity component corresponding to the current predicted position via finite differencing.
	v_t = (q_t - q_{t-1})/dt

	Inputs:
	initial_states: shape = [tw, batch_size( =1), 13] Represents the input to the network in the current step.
	output: shape = [batch_size( =1), 7] Represents the output predicted by the network for initial_states.

	Output:
	full_pred_state: *numpy array* of shape = [1,13] composing of (q_pred, v_{finite_diff})
	"""
	last_state = initial_states[-1,0,:]
	# output[0,3:7] = output[0,3:7]/np.linalg.norm(output[0,3:7].numpy())
	quat_prev = last_state[3:7].numpy()
	quat_prev = np.array([quat_prev[1], quat_prev[2], quat_prev[3], quat_prev[0]])/np.linalg.norm(quat_prev)
	quat_curr = output[0,3:7].numpy()
	quat_curr = np.array([quat_curr[1], quat_curr[2], quat_curr[3], quat_curr[0]])/np.linalg.norm(quat_curr)
	dw_t = (get_angular_velocity(quat_prev.reshape(1,-1), quat_curr.reshape(1, -1), dt))

	pos_prev = last_state[:3].numpy()
	pos_curr = output[0,:3].numpy()
	d_t = (pos_curr - pos_prev)/dt

	full_pred_vel = np.concatenate((d_t, dw_t))

	full_pred_state = (np.concatenate((output[0,:], full_pred_vel))).reshape(1,-1)

	return full_pred_state

def get_angular_pos(quat_prev, angVel_avg, dt = 0.0067):
	w_norm = np.linalg.norm(angVel_avg)
	theta = w_norm*dt
	axis_ = angVel_avg/w_norm
	quat_rot = np.array([axis_[0]*np.sin(theta/2), axis_[1]*np.sin(theta/2), axis_[2]*np.sin(theta/2), np.cos(theta/2)])
	rot = R.from_quat(quat_rot)

	# rot = angVel_avg*dt
	# rot = R.from_euler("XYZ", rot)

	prev_rot = R.from_quat(quat_prev)
	res_rot = prev_rot*rot
	res_rot_quat = res_rot.as_quat()
	res_rot_quat = np.array([res_rot_quat[-1], res_rot_quat[0], res_rot_quat[1], res_rot_quat[2]])

	# print("res_rot_quat: ", res_rot_quat)
	return res_rot_quat

def compute_state_pos(initial_states, output, dt = 0.0067):
	"""
	This method computed the position component corresponding to the current predicted velocity via finite differencing.
	q_t = q_{t-1} + (0.5*v_{pred} + 0.5*v_{t-1})*dt

	Inputs:
	initial_states: shape = [tw, batch_size( =1), 13] representing the input to the network in current step
	output: shape = [batch_size ( =1), 6] represents the velocity predicted for the current timestep.

	Output:
	full_pred_state: *numpy array* of shape = [1, 13] composing of [q_{finite_diff}, v_pred]
	"""

	last_state = initial_states[-1,0,:]
	#TODO: if pred_mode == "diff_vel": add last_state to output

	angVel_prev = last_state[10:].numpy()
	angVel_curr = output[0,3:].numpy()
	angVel_avg = 0.5*(angVel_prev + angVel_curr)
	quat_prev = last_state[3:7].numpy()
	quat_prev = np.array([quat_prev[1], quat_prev[2], quat_prev[3], quat_prev[0]])
	quat_curr = get_angular_pos(quat_prev, angVel_avg, dt = 0.0067)

	linearVel_prev = last_state[7:10].numpy()
	linearVel_curr = output[0,:3].numpy()
	linearVel_avg = 0.5*(linearVel_prev + linearVel_curr)
	pos_prev = last_state[:3].numpy()
	pos_curr = pos_prev + (linearVel_avg*dt)

	full_pred_pos = np.concatenate((pos_curr, quat_curr))
	# print("full_pred_pos: ", full_pred_pos)

	full_pred_state = np.concatenate((full_pred_pos, output[0,:])).reshape(1,-1)
	# print("full_pred_state: ", full_pred_state)

	return full_pred_state

def rollout_trajectory(trained_model, loss_function, state_sequence,  state_labels,tw = 16, time_steps = 200, pred_mode = "vel", \
						scale_mean = 0, scale_std = 1, verbose = False, weighted_loss = False, weights = [1,1], gt_trajectory = None):
	"""
	This method rolls out the trajectory for N=`time_steps` time.

	Inputs:
	trained_model: Object of the model trained
	state_sequence: numpy array of shape = [tw, 13] corresponding to the initial states from which the trajectory is to be rolled out
	state_labels: numpy array of shape (time_steps, 13) giving the ground truth corresponding to each time-step
	time_steps: number of time_steps for which trajectory is supposed to be rolled out.

	Outputs:
	trajectory_: numpy array of shape = [tw+time_steps, 13] giving the trajectory roll-out
	"""
	total_err_theta = 0
	total_err_pos = 0
	diff_theta = []
	loss_items = []
	time_steps = np.minimum(state_labels.shape[0], time_steps)
	# print("time-steps: ", time_steps)
	device = next(trained_model.parameters()).device
	if device is not torch.device("cpu"):
		trained_model.to(torch.device("cpu"))
	# scale_mean, scale_std = scale_mean.to(device), scale_std.to(device)
	# pdb.set_trace()
	trained_model.eval()
	trajectory_ = np.copy(state_sequence)
	input_sequence = torch.from_numpy(np.expand_dims(state_sequence, axis = 1)).float()
	state_labels = torch.from_numpy(state_labels).float()
	if gt_trajectory is not None:
		gt_trajectory = torch.from_numpy(gt_trajectory).float()
	rolled_time = -1
	running_loss = 0
	with torch.no_grad():
		while(rolled_time < time_steps - 1):
			rolled_time += 1
			output = trained_model((input_sequence - scale_mean)/scale_std)
			if pred_mode == "pos":
				loss = loss_function(output, state_labels[rolled_time, :7].reshape(1,-1))
				full_pred_state = compute_state_vel(input_sequence, output)
				err_pos, err_theta = getRotationalError(input_sequence, full_pred_state, state_labels, rolled_time, tw)
			elif pred_mode == "vel":
				positional_loss = loss_function(output[:,:3], state_labels[rolled_time,7:10].reshape(1,-1))
				rotational_loss = loss_function(output[:,3:], state_labels[rolled_time,10:].reshape(1,-1))
				if weighted_loss:
					loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
				else:
					loss = loss_function(output, state_labels[rolled_time, 7:].reshape(1,-1))
				full_pred_state = compute_state_pos(input_sequence, output)
				err_pos, err_theta = getRotationalError(input_sequence, full_pred_state, state_labels, rolled_time, tw)
			elif pred_mode == "full":
				loss = loss_function(output, state_labels[rolled_time, :].reshape(1,-1))
				full_pred_state = output.numpy()
				err_pos, err_theta = getRotationalError(input_sequence, full_pred_state, state_labels, rolled_time, tw)
			elif pred_mode == "dvel":
				positional_loss = loss_function(output[:,:3], state_labels[rolled_time,7:10].reshape(1,-1))
				rotational_loss = loss_function(output[:,3:], state_labels[rolled_time,10:].reshape(1,-1))
				if weighted_loss:
					loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
				else:
					loss = loss_function(output, state_labels[rolled_time, 7:].reshape(1,-1))
				output += input_sequence[-1,0,7:]
				full_pred_state = compute_state_pos(input_sequence, output)
				err_pos, err_theta = getRotationalError(input_sequence, full_pred_state, gt_trajectory, rolled_time, tw)
			elif pred_mode == "dpos":
				positional_loss = loss_function(output[:,:3], state_labels[rolled_time,:3].reshape(1,-1))
				rotational_loss = loss_function(output[:,3:], state_labels[rolled_time,3:7].reshape(1,-1))
				if weighted_loss:
					loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
				else:
					loss = loss_function(output, state_labels[rolled_time, :7].reshape(1,-1))
				output += input_sequence[-1,0,:7]
				full_pred_state = compute_state_vel(input_sequence, output)
				err_pos, err_theta = getRotationalError(input_sequence, full_pred_state, gt_trajectory, rolled_time, tw)
			elif pred_mode == "dx":
				if weighted_loss:
					print("Weighted loss not supported in dx!")
					sys.exit(1)
				else:
					loss = loss_function(output, state_labels[rolled_time,:].reshape(1,-1))
				output += input_sequence[-1,0,:]
				output[0,3:7] = output[0,3:7]/np.linalg.norm(output[0,3:7])
				full_pred_state = output
				err_pos, err_theta = getRotationalError(input_sequence, full_pred_state, gt_trajectory, rolled_time, tw)
			elif pred_mode == "fs":
				if weighted_loss:
					print("Weighted loss not supported in dx!")
					sys.exit(1)
				else:
					loss = loss_function(output, state_labels[rolled_time, :7].reshape(1,-1))
				output[0,3:7] = output[0,3:7]/np.linalg.norm(output[0,3:7])
				full_pred_state = np.zeros((1,13))
				full_pred_state[0,:7] = output
				# pdb.set_trace()
				err_pos, err_theta = getRotationalError(input_sequence, full_pred_state, state_labels, rolled_time, tw)
			loss_items.append(loss.item())
			running_loss += loss.item();
			total_err_pos += err_pos
			total_err_theta += err_theta
			diff_theta.append(err_theta)
			if verbose:
				print("input_sequence: ", input_sequence)
				print("output: ", output)
				print("full_pred_state: ", full_pred_state)
				print("--------------------------------------------------------------")
			trajectory_ = np.append(trajectory_, full_pred_state, axis = 0)
			input_sequence = (torch.from_numpy(np.concatenate((input_sequence[1:,], \
		                                    np.expand_dims(full_pred_state, axis = 1)), axis = 0))).float()
	if (rolled_time == 0): rolled_time += 1
	return trajectory_, running_loss/rolled_time, total_err_pos/rolled_time, total_err_theta/rolled_time, diff_theta

def pred_with_gt(rollout_dataloader, model, loss_function, tw = 16, pred_mode = "vel", plot_graphs = True, verbose = False, scale_mean = 0, scale_std = 1, weighted_loss = False, weights = [1,1]):

	num_states = 13
	start_ = 0
	plot_labels = np.empty((0,num_states))
	plot_predicted = np.empty((0,num_states))
	device = next(model.parameters()).device
	if device is not torch.device("cpu"):
		model.to(torch.device("cpu"))
	# scale_mean, scale_std = scale_mean.to(device), scale_std.to(device)
	model.eval()
	running_loss = 0
	overall_step = 0
	total_err_rot = 0
	total_err_pos = 0
	total_err_theta = 0
	total = 0
	with torch.no_grad():
		for i, (data, labels) in enumerate(rollout_dataloader):
			input_ = np.transpose(data, (1,0,2))
			# input_ = input_ - scale_mean/scale_std
			output = model((input_ - scale_mean)/scale_std)
			if pred_mode == "vel":
				positional_loss = loss_function(output[:,:3], labels[:,7:10])
				rotational_loss = loss_function(output[:,3:], labels[:,10:])
				if weighted_loss:
					loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
				else:
					loss = loss_function(output, labels[:, 7:].reshape(1,-1))
				full_pred_state = compute_state_pos(input_, output)
				err_pos, err_theta = getRotationalError(input_, full_pred_state, labels, single_step = True)
				plot_labels = np.append(plot_labels, labels.numpy(), axis = 0)
				plot_predicted = np.append(plot_predicted, full_pred_state, axis = 0)
			elif pred_mode == "pos":
				loss = loss_function(output, labels[:,:7])
				full_pred_state = compute_state_vel(input_, output)
				err_pos, err_theta = getRotationalError(input_, full_pred_state, labels, single_step = True)
				plot_labels = np.append(plot_labels, labels.numpy(), axis = 0)
				plot_predicted = np.append(plot_predicted, full_pred_state, axis = 0)
			elif pred_mode == "full":
				loss = loss_function(output, labels)
				full_pred_state = output.numpy()
				err_pos, err_theta = getRotationalError(input_, full_pred_state, labels, single_step = True)
				plot_labels = np.append(plot_labels, labels.numpy(), axis = 0)
				plot_predicted = np.append(plot_predicted, full_pred_state, axis = 0)
			elif pred_mode == "dvel":
				positional_loss = loss_function(output[:,:3], labels[:,7:10])
				rotational_loss = loss_function(output[:,3:], labels[:,10:])
				if weighted_loss:
					loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
				else:
					loss = loss_function(output, labels[:, 7:].reshape(1,-1))
				output += input_[-1,0,7:]
				full_pred_state = compute_state_pos(input_, output)
				err_pos, err_theta = getRotationalError(input_, full_pred_state, labels + input_[-1], single_step = True)
				plot_labels = np.append(plot_labels, labels.numpy(), axis = 0)
				plot_predicted = np.append(plot_predicted, full_pred_state, axis = 0)
			elif pred_mode == "dpos":
				positional_loss = loss_function(output[:,:3], labels[:,:3])
				rotational_loss = loss_function(output[:,3:], labels[:,3:7])
				if weighted_loss:
					loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
				else:
					loss = loss_function(output, labels[:, :7].reshape(1,-1))
				output += input_[-1,0,:7]
				full_pred_state = compute_state_vel(input_, output)
				err_pos, err_theta = getRotationalError(input_, full_pred_state, labels + input_[-1], single_step = True)
				plot_labels = np.append(plot_labels, labels.numpy(), axis = 0)
				plot_predicted = np.append(plot_predicted, full_pred_state, axis = 0)
			elif pred_mode == "dx":
				positional_loss = loss_function(output[:,:3], labels[:,:3]) + loss_function(output[:,7:10], labels[:,7:10])
				rotational_loss = loss_function(output[:,3:7], labels[:,3:7]) + loss_function(output[:,10:], labels[:,10:])
				if weighted_loss:
					loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
				else:
					loss = loss_function(output, labels[:, :].reshape(1,-1))
				output += input_[-1,0,:]
				output[0,3:7] = output[0,3:7]/np.linalg.norm(output[0,3:7])
				full_pred_state = output
				err_pos, err_theta = getRotationalError(input_, full_pred_state, labels + input_[-1], single_step = True)
				plot_labels = np.append(plot_labels, labels.numpy(), axis = 0)
				plot_predicted = np.append(plot_predicted, full_pred_state, axis = 0)
			elif pred_mode == "fs":
				if weighted_loss:
					loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
				else:
					loss = loss_function(output, labels[:, :7].reshape(1,-1))
				output[0,3:7] = output[0,3:7]/np.linalg.norm(output[0,3:7])
				full_pred_state = np.zeros((1,13))
				full_pred_state[0,:7] = output
				err_pos, err_theta = getRotationalError(input_, full_pred_state, labels, single_step = True)				
			if verbose:
				print("i: ", i)
				print("input: ", input_)
				print("output: ", output)
				print("labels: ", labels[:,:])
				print("full pred state: ", full_pred_state)
				print("-------------------------------")
			running_loss += loss.item()
			total_err_pos += err_pos
			total_err_theta += err_theta
			overall_step += 1
			total += labels.shape[0]

	y_labels_ = ["x", "y", "z", "quat_w", "quat_x", "quat_y", "quat_z", "v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]
	if plot_graphs:
		for j in range(start_, start_ + num_states):
			plt.plot(plot_labels[:,j-start_],"r", label = "actual")
			plt.plot(plot_predicted[:,j-start_],"b", label = "predicted")
			plt.xlabel("time-step")
			plt.ylabel(y_labels_[j])
			plt.legend()
			plt.show()

	return running_loss/total, total_err_pos/total, total_err_theta/total

def loadModel(PATH, trained_model, model_name = None):
	trained_model_path = os.path.join(PATH, model_name)
	device = torch.device("cpu")
	checkpoint = torch.load(trained_model_path + ".pt", map_location = device)
	# trained_model.load_state_dict(torch.load(trained_model_path + ".pt", map_location=device))
	trained_model.load_state_dict(checkpoint['model_state_dict'])
	trained_model.eval()

	return trained_model

def evaluateRollout(PATH, trained_model, loss_function, num_tosses = 100, toss_start = 1, pred_mode = "vel", isTest = False, tw = 16, mean_ = 0, std_ =  1, weighted_loss = False, weights = [1,1]):
	if weighted_loss:
		print("Inside evaluate Rollout, using weighted loss with weights: ", weights)

	start =  toss_start if not isTest else 10001
	total_loss = 0
	total_err_pos = 0
	total_err_theta = 0
	for i in range(num_tosses):
		test_data_ = np.loadtxt(PATH+str(start+i)+".csv", delimiter = ',').T
		state_sequence = test_data_[:tw] #(tw, 13)
		if (pred_mode == "dvel" or pred_mode == "dpos" or pred_mode == "dx"):
			state_labels = test_data_[tw:,:] - test_data_[tw-1:-1,:]  #check this
		elif pred_mode == "fs":
			state_labels = test_data_[-1:]
		else:
			state_labels = test_data_[tw:] #shape = (x, 13)
		ground_truth_trajectory = test_data_[tw:] 
		trajectory_, loss, err_pos, err_theta, _ = rollout_trajectory(trained_model, loss_function, state_sequence, state_labels, \
													tw = tw, time_steps = 50, pred_mode=pred_mode, scale_mean = (mean_), scale_std = (std_), weighted_loss = weighted_loss, weights = weights, gt_trajectory = ground_truth_trajectory)
		total_loss += loss
		total_err_pos += err_pos
		total_err_theta += err_theta
		
	ROLLOUT_ROT_ERR = total_err_theta/num_tosses
	ROLLOUT_LOSS = total_loss/num_tosses
	ROLLOUT_POS_ERR = total_err_pos/num_tosses

	return ROLLOUT_LOSS, ROLLOUT_ROT_ERR, ROLLOUT_POS_ERR

def evaluateSinglestep(PATH, trained_model, loss_function, num_tosses = 100, toss_start = 1, pred_mode = "vel", isTest = False, tw = 16, mean_ = 0, std_ =  1, weighted_loss = False, weights = [1,1]):
	if weighted_loss:
		print("Inside evaluate singlestep, using weighted loss with weights: ", weights)	
	start = toss_start if not isTest else 10001
	total_loss = 0
	total_err_pos = 0
	total_err_theta = 0
	for i in range(num_tosses):
		X_roll, y_roll = learning_utils.getRecurrentDataset(PATH, start+i, 1, tw=tw, isEval = True, predict_mode = pred_mode)
		rollout_dataset = CubeTossDataset(X_roll,y_roll)
		rollout_dataloader = DataLoader(rollout_dataset, batch_size = 1, shuffle = False)
		loss, err_pos, err_theta = pred_with_gt(rollout_dataloader, trained_model, loss_function, pred_mode = pred_mode, plot_graphs = False, \
		                                      scale_mean = (mean_), scale_std = (std_), \
		                                      verbose = False, weighted_loss = weighted_loss, weights = weights)
		total_loss += loss
		total_err_pos += err_pos
		total_err_theta += err_theta
  # print("--------------------------------")

	SINGLESTEP_ROT_ERR = total_err_theta/num_tosses
	SINGLESTEP_LOSS = total_loss/num_tosses
	SINGLESTEP_POS_ERR = total_err_pos/num_tosses

	# print("Average train singlestep loss: ", total_loss/num_tosses)
	# print("Average postional error: ", total_err_pos/num_tosses)
	# print("Average rotational error: ", total_err_theta/num_tosses)
	return SINGLESTEP_LOSS, SINGLESTEP_ROT_ERR, SINGLESTEP_POS_ERR

if __name__ == "__main__":
	print("eval_utils.py executed!")