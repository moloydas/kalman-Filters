import numpy as np
import math
from numpy.linalg import inv,pinv
import matplotlib.pyplot as plt
import scipy.linalg
import time

#load dataset
dataset = np.load("dataset.npz")
t=dataset['t']
x_true=dataset['x_true']
y_true=dataset['y_true']
th_true=dataset['th_true']
l=dataset['l']
r=dataset['r']
b=dataset['b']
vel=dataset['v']
om=dataset['om']
d=dataset['d'][0][0]
dt=0.1


# Given variances
r_var=dataset['r_var'][0][0]
b_var=dataset['b_var']
v_var=dataset['v_var'][0][0]
om_var=dataset['om_var'][0][0]

'''
v_var = 0.004420255225				sd = 0.06633
om_var = 0.008186087529				sd = 0.09047
r_var = 0.0009003600360000001		sd = 0.03
b_var = [[0.00067143]]				sd = 0.025911966
'''

# v_var = 0.9
# om_var = 0.5
# r_var = 0.01
# b_var = 0.04

param = {'alpha':1e-3,
        'beta':  2,
        'kappa': 1,
        'dims':  3}

Q = np.eye(3)
Q[0][0] = 0.0009
Q[1][1] = 0.0009
Q[2][2] = 0.0008
R = np.array([ [0.00095,0], [0,0.0008] ], dtype=np.double)

def circular_limit(theta_val):

	if theta_val.shape:
		n = theta_val.shape[0]
		for i in range(n):
			if theta_val[i] > math.pi:
				theta_val[i] = -1 * (2*math.pi - theta_val[i])
			elif theta_val[i] < -math.pi:
				theta_val[i] = -1 * (-2*math.pi - theta_val[i])
	else:
		if theta_val> math.pi:
			theta_val = -1 * (2*math.pi - theta_val)
		elif theta_val< -math.pi:
			theta_val = -1 * (-2*math.pi - theta_val)
		else:
			theta_val = theta_val

	return theta_val

def calculate_sigma_points(state, covariance, param ):
	scaling_factor = ((param['alpha']**2) * (param['dims'] + param['kappa'])) - param['dims']
	sigma_points = np.zeros((param['dims'], (2*param['dims'] + 1) ),dtype = np.double)
	sigma = scipy.linalg.sqrtm(covariance)
	sigma = np.sqrt(param['dims'] + scaling_factor) * sigma
	sigma_points[:,0] = state[:,0]
	sigma_points[:, 1:(param['dims']+1) ] = state + sigma
	sigma_points[:, (param['dims']+1): ] = state - sigma
	return sigma_points

def generate_weights(param):
	scaling_factor = ((param['alpha']**2) * (param['dims'] + param['kappa'])) - param['dims']
	W_s = np.zeros((1,(2*param['dims'] + 1)),dtype=np.double)
	W_c = np.zeros((1,(2*param['dims'] + 1)),dtype=np.double)
	W_s[0,0] = scaling_factor/(param['dims'] + scaling_factor)
	W_c[0,0] = (scaling_factor/(param['dims'] + scaling_factor)) + ( 1 - param['alpha']**2 + param['beta'] )
	W_s[0,1:] = 1/(2*(param['dims'] + scaling_factor))
	W_c[0,1:] = 1/(2*(param['dims'] + scaling_factor))
	return W_s,W_c

def propagate_state(state_prev, v, w):
	n = state_prev.shape
	state_pred = np.zeros(state_prev.shape,dtype=np.double)
	state_pred[0,:] = state_prev[0,:] + 0.1 * v * np.cos(state_prev[2,:])
	state_pred[1,:] = state_prev[1,:] + 0.1 * v * np.sin(state_prev[2,:])
	# state_pred[2,:] = circular_limit(state_prev[2,:] + 0.1 * w)
	state_pred[2,:] = (state_prev[2,:] + 0.1 * w)
	if param['dims'] > 3:
		state_pred[3:,:] = state_prev[3:,:]
	return state_pred

def calculate_z_pred(x_pred, l):
	n = x_pred.shape
	z_pred = np.zeros((2,n[1]),dtype='float')

	# measured range and angle of landmark
	a = l[0] - x_pred[0,:] - d*np.cos(x_pred[2,:])
	b = l[1] - x_pred[1,:] - d*np.sin(x_pred[2,:])
	
	r = np.sqrt( np.square(a)+np.square(b) )
	phi = r.copy()
	for i in range(n[1]):
		phi[i] = math.atan2(b[i], a[i]) - x_pred[2,i]
	z_pred[0,:] = r
	# z_pred[1,:] = circular_limit(phi)
	z_pred[1,:] = (phi)

	return z_pred

def run_ukf(x_prev,P_prev,v,w,l,z,param,prop_flag):
	global pred_fuckup

	# calculate sigma points
	sigma_points = calculate_sigma_points(x_prev, P_prev, param)

	if prop_flag == 1:
		# propagate state
		x_pred = propagate_state(sigma_points,v,w)

		# calculate weights
		W_s, W_c = generate_weights(param)

		# predicted state
		x_pred_mu = np.sum(W_s * x_pred,axis=1).reshape(param['dims'],1)
		# x_pred_mu[2,0] = circular_limit(x_pred_mu[2,0])
		# print("x_pred:\n",x_pred)
		# print("x_pred_mu: \n", x_pred_mu)

		# update covariance
		P_pred = np.zeros((param['dims'],param['dims']),dtype=np.double)

		for i in range(2*param['dims'] + 1):
			A = (x_pred[:,i].reshape(-1,1) - x_pred_mu[:,:])
			P = A @ A.T
			P = W_c[0,i] * P
			P_pred += P

		P_pred = P_pred + Q
		# print("P_pred: \n",P_pred)
	else:
		x_pred = sigma_points
		x_pred_mu = x_prev
		P_pred = P_prev

	# calculate z_pred
	z_pred = calculate_z_pred(x_pred, l)

	# calculate weights
	W_s, W_c = generate_weights(param)

	# predicted state
	z_pred_mu = np.sum(W_s * z_pred,axis=1).reshape(2,1)
	temp = z_pred_mu[1,0]
	# z_pred_mu[1,0] = circular_limit(z_pred_mu[1,0])
	# if(temp != z_pred_mu[1,0]):
	# 	print("predicting a fuckup!!!")
	# 	print("old: ",temp, " new: ",z_pred_mu[1,0])

	# print("z_pred_mu:\n",z_pred_mu)
	# print("z_pred:\n",z_pred)

	# update covariance
	P_z_pred = np.zeros((2,2),dtype=np.double)

	for i in range(2*param['dims'] + 1):
		A = (z_pred[:,i].reshape(-1,1) - z_pred_mu[:,:])
		P = A @ A.T
		P = W_c[0,i] * P
		P_z_pred += P
	
	P_z_pred = P_z_pred + R
	# print("P_z_pred: \n", P_z_pred)
	P_x_z_pred = np.zeros((param['dims'],2),dtype=np.double)

	for i in range(2*param['dims'] + 1):
		A = (x_pred[:,i].reshape(-1,1) - x_pred_mu[:,:])
		B = (z_pred[:,i].reshape(-1,1) - z_pred_mu[:,:])
		P = A @ B.T
		P = W_c[0,i] * P
		P_x_z_pred += P

	# update kalman gain
	kalman_gain =  P_x_z_pred @ inv(P_z_pred)
	# print("P_x_z_pred: \n", P_x_z_pred)
	# print("P_z_pred: \n", P_z_pred)
	# print("inv(P_z_pred): \n", inv(P_z_pred))
	# print("kalman gain: ", kalman_gain)

	# apply circ limit
	x_pred_mu[2,0] = circular_limit(x_pred_mu[2,0])
	z_pred_mu[1,0] = circular_limit(z_pred_mu[1,0])

    # Correct State
	if(z[1,0] * z_pred_mu[1,0] < 0):
		# print("predicting a fuckup!!!")
		# print("measure: ",z[1,0], " predicted: ",z_pred_mu[1,0])
		# print("kalman gain: \n",kalman_gain)
		# print("x_pred_mu:\n",x_pred_mu)
		pred_fuckup += 1
	x_est = x_pred_mu + kalman_gain @ (z - z_pred_mu) ### TO DO:: handle this case z = 3.1 z_pred = -3.1
	x_est[2,0] = circular_limit(x_est[2,0])
	P_est = P_pred - kalman_gain @ P_z_pred @ kalman_gain.T

	return x_est,P_est

# init x and P
x_prev = np.array([x_true[0],y_true[0],th_true[0]],dtype=np.double)
P_prev = np.array([[1,0,0],[0,1,0],[0,0,.5]],dtype=np.double)

z = np.zeros((2,1),dtype=np.double)

use_only_one_sensor = 1
fuck_cntr_theta = 0
fuck_cntr_y = 0
fuck_cntr_x = 0
pred_fuckup = 0

start_time = 1
end_time = len(x_true)
offset = 0

if __name__ == '__main__':
	for j in range(start_time,end_time):
		v = vel[j]
		w = om[j]
		# print("\niter: " + str(j))
		prop_flag = 1
		for i in range(0,len(l)):
			if r[j,i] != 0:
				z[0,0] = r[j,i]
				z[1,0] = b[j,i]
				landmark = l[i,:]

				x_est,P_est = run_ukf(x_prev, P_prev, v, w, landmark, z, param,prop_flag)
				prop_flag = 1
				x_prev = x_est
				P_prev = P_est

				prop_flag = 0

		error_x = x_est[0,0] - x_true[j]
		error_y = x_est[1,0] - y_true[j]
		error_theta = x_est[2,0] - th_true[j]

		if abs(error_theta) > 1e1:
			print("fuck fuck !!")
			fuck_cntr_theta += 1
			break
		elif abs(error_y) > 1e1:
			print("fuck fuck !!")
			fuck_cntr_y += 1
		elif abs(error_x) > 1e1:
			print("fuck fuck !!")
			fuck_cntr_x += 1

		np.set_printoptions(precision=5)

		if j%1000 == 0:
			print("\niter: " + str(j))
			print(x_est)
			print(P_est)
			print("x_error:" + str(error_x))
			print("y_error:" + str(error_y))
			print("theta_error:" + str(error_theta), "true_val" + str(th_true[j]))
		if j==start_time:
			final_mu=x_est.T
			final_error_x = error_x
			final_error_y = error_y
			final_error_theta = error_theta
		else:
			final_mu = np.concatenate((final_mu,x_est.T))
			final_error_x = np.concatenate((final_error_x,error_x))
			final_error_y = np.concatenate((final_error_y,error_y))
			final_error_theta = np.concatenate((final_error_theta,error_theta))

		if j%1000 == 0:
			plt.plot(l[:,0],l[:,1],'ro')
			plt.plot(landmark[0],landmark[1],'go')
			plt.plot(x_true[offset:j,0], y_true[offset:j,0],'yo')
			plt.plot(final_mu[offset:j,0], final_mu[offset:j,1],'bo')
			plt.pause(0.05)

	print("\naverage x error: " + str(np.sqrt(np.mean(final_error_x**2))))
	print("max error x: ", np.max(final_error_x))
	print("average y error: " + str(np.sqrt(np.mean(final_error_y**2))))
	print("max error y: ", np.max(final_error_y))
	print("average theta error: " + str(np.sqrt(np.mean(final_error_theta**2))))
	print("max error theta: ", np.max(final_error_theta))
	print("fuck cntr: ", fuck_cntr_theta," y:",fuck_cntr_y," x:",fuck_cntr_x, "pred_fuckup: ", pred_fuckup)

	#ground thruth plot
	#plt.plot(l[:,0],l[:,1],'ro')
	#plt.plot(x_true, y_true, 'ro')
	#plt.show()

	#predictions 
	# plt.plot(l[:,0],l[:,1],'ro')
	# plt.plot(final_mu[:,0], final_mu[:,1],'bo')
	plt.show()
