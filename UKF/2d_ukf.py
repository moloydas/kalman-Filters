import numpy as np
import math
from numpy.linalg import inv,pinv
import matplotlib.pyplot as plt
import scipy.linalg
import time

def circular_limit(theta_val):

	if theta_val.shape:
		n = theta_val.shape[0]
		theta_val[theta_val > math.pi] = -1 * (2*math.pi - theta_val[theta_val > math.pi])
		theta_val[theta_val < -math.pi] = -1 * (-2*math.pi - theta_val[theta_val < -math.pi])
	else:
		if theta_val> math.pi:
			theta_val = -1 * (2*math.pi - theta_val)
		elif theta_val< -math.pi:
			theta_val = -1 * (-2*math.pi - theta_val)

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
	z_pred[1,:] = (phi)

	return z_pred

def predicted_state(x_prev_mu, P_prev, v, w, param):
    # calculate sigma points
	sigma_points = calculate_sigma_points(x_prev_mu, P_prev, param)

    # propagate state
	x_pred = propagate_state(sigma_points,v,w)

	# calculate weights
	W_s, W_c = generate_weights(param)

	# predicted state
	x_pred_mu = np.sum(W_s * x_pred,axis=1).reshape(param['dims'],1)

	# update covariance
	P_pred = np.zeros((param['dims'],param['dims']),dtype=np.double)
	A = (x_pred - x_pred_mu)
	P_pred = (W_c * A) @ A.T
	P_pred = P_pred + Q

	return x_pred_mu,P_pred

def correct_state(x_pred_mu, P_pred, z, l, param):

	# calculate sigma points
	x_pred = calculate_sigma_points(x_pred_mu, P_prev, param)

	# calculate z_pred
	z_pred = calculate_z_pred(x_pred, l)

	# calculate weights
	W_s, W_c = generate_weights(param)

	# predicted state
	z_pred_mu = np.sum(W_s * z_pred,axis=1).reshape(2,1)

	# Calculate measurement covariance
	P_z_pred = np.zeros((2,2),dtype=np.double)
	A = (z_pred - z_pred_mu)
	P_z_pred = (W_c * A) @ A.T
	P_z_pred = P_z_pred + R

	# Calculate cross covariance 
	P_x_z_pred = np.zeros((param['dims'],2),dtype=np.double)
	A = (x_pred - x_pred_mu)
	B = (z_pred - z_pred_mu)
	P_x_z_pred = (W_c * A) @ B.T

	# update kalman gain
	kalman_gain =  P_x_z_pred @ inv(P_z_pred)

	# apply circ limit
	x_pred_mu[2,0] = circular_limit(x_pred_mu[2,0])
	z_pred_mu[1,0] = circular_limit(z_pred_mu[1,0])

	# Correct State
	x_est = x_pred_mu + kalman_gain @ (z - z_pred_mu)
	x_est[2,0] = circular_limit(x_est[2,0])
	P_est = P_pred - kalman_gain @ P_z_pred @ kalman_gain.T

	return x_est, P_est

np.set_printoptions(precision=5)

if __name__ == '__main__':
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

	param = {'alpha':1e-3,
			'beta':  2,
			'kappa': 1,
			'dims':  3}

	Q = np.eye(3)
	Q[0][0] = 0.0004
	Q[1][1] = 0.0004
	Q[2][2] = 0.0008
	R = np.array([ [0.00095,0], [0,0.0008] ], dtype=np.double)

	# init x and P
	x_prev = np.array([x_true[0],y_true[0],th_true[0]],dtype=np.double)
	P_prev = np.array([[1,0,0],[0,1,0],[0,0,.1]],dtype=np.double)

	z = np.zeros((2,1),dtype=np.double)

	start_time = 1
	end_time = len(x_true)
	offset = 0

	final_mu= np.zeros((len(x_true),3))
	error 	= np.zeros((len(x_true),3))

	for j in range(start_time,end_time):
		v = vel[j]
		w = om[j]

		x_pred_mu, P_pred = predicted_state(x_prev, P_prev, v, w, param)
		x_est = x_pred_mu
		P_est = P_pred

		for i in range(1,len(l)):
			if r[j,i] != 0:
				z[0,0] = r[j,i]
				z[1,0] = b[j,i]
				landmark = l[i,:]
				x_est,P_est = correct_state(x_est, P_est, z, landmark, param)

		x_prev = x_est
		P_prev = P_est

		final_mu[j,:] = np.ravel(x_est.T)
		error[j,0] = x_est[0,0] - x_true[j]
		error[j,1] = x_est[1,0] - y_true[j]
		error[j,2] = abs(x_est[2,0]) - abs(th_true[j])		# ignoring error sign

		if j%1000 == 0:
			print("\niter: " + str(j))
			print(x_est)
			print(P_est)
			print("x_error:" + str(error[j,0]))
			print("y_error:" + str(error[j,1]))
			print("theta_error:" + str(error[j,2]), "true_val" + str(th_true[j]))

		# if j%1000 == 0:
		# 	plt.plot(l[:,0], l[:,1],'ro')
		# 	plt.plot(l[0], l[1],'go')
		# 	plt.plot(x_true[offset:j,0], y_true[offset:j,0],'yo')
		# 	plt.plot(final_mu[offset:j,0], final_mu[offset:j,1],'bo')
		# 	plt.pause(0.05)

	print("\naverage x error: " + str(np.sqrt(np.mean(error[:,0]**2))))
	print("max error x: ", np.max(abs(error[:,0])))
	print("average y error: " + str(np.sqrt(np.mean(error[:,1]**2))))
	print("max error y: ", np.max(abs(error[:,1])))
	print("average theta error: " + str(np.sqrt(np.mean(error[:,2]**2))))
	print("max error theta: ", np.max(abs(error[:,2])))

	plt.plot(l[:,0], l[:,1],'ro')
	plt.plot(x_true[:,0], y_true[:,0],'yo')
	plt.plot(final_mu[:,0], final_mu[:,1],'bo')
	plt.show()
	plt.plot(error[:,0])
	plt.show()
	plt.plot(error[:,1])
	plt.show()
	plt.plot(error[:,2])
	plt.show()
