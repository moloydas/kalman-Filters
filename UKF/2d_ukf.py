import numpy as np
import math
from numpy.linalg import inv,pinv
import matplotlib.pyplot as plt
import scipy.linalg

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

'''
# Given variances
r_var=dataset['r_var'][0][0]
b_var=dataset['b_var']
v_var=dataset['v_var'][0][0]
om_var=dataset['om_var'][0][0]
'''
'''
v_var = 0.004420255225				sd = 0.06633
om_var = 0.008186087529				sd = 0.09047
r_var = 0.0009003600360000001		sd = 0.03
b_var = [[0.00067143]]				sd = 0.025911966
'''

v_var = 0.9
om_var = 0.5
r_var = 0.01
b_var = 0.04

param = {'alpha':1e-3,
        'beta':  2,
        'kappa': 1,
        'dims':  5}

Q = np.array([ [v_var,0], [0,om_var] ],dtype=np.double)
R = np.array([ [r_var,0], [0,b_var] ], dtype=np.double)
V = np.eye(2, dtype=np.double)

def circular_limit(theta_val):
	i = 0
	new_theta = np.zeros(theta_val.shape,dtype=np.double)
	for theta in theta_val:
		if theta > math.pi:
			new_theta[i] = -1 * (2*math.pi - theta)
		elif theta < -math.pi:
			new_theta[i] = -1 * (-2*math.pi - theta)
		else:
			new_theta[i] = theta
		i = i + 1
	return new_theta

def calculate_sigma_points(state, covariance, param ):
	scaling_factor = ((param['alpha']**2) * (param['dims'] + param['kappa'])) - param['dims']
	sigma_points = np.zeros((param['dims'], (2*param['dims'] + 1) ),dtype = np.double)
	sigma = scipy.linalg.sqrtm((param['dims'] + scaling_factor) * covariance)
	sigma_points[:,0] = state[:,0]
	sigma_points[:,1:6] = state + sigma
	sigma_points[:,6:11] = state - sigma
	return sigma_points

def generate_weights(param):
	scaling_factor = ((param['alpha']**2) * (param['dims'] + param['kappa'])) - param['dims']
	W_s = np.zeros((1,11),dtype=np.double)
	W_c = np.zeros((1,11),dtype=np.double)
	W_s[0,0] = scaling_factor/(param['dims'] + scaling_factor)
	W_c[0,0] = (scaling_factor/(param['dims'] + scaling_factor)) + ( 1 - param['alpha']**2 + param['beta'] )
	W_s[0,1:] = 1/(2*(param['dims'] + scaling_factor))
	W_c[0,1:] = 1/(2*(param['dims'] + scaling_factor))
	return W_s,W_c

def propagate_state(state_prev, v, w):
    n = state_prev.shape
    state_pred = np.zeros((5,n[1]),dtype=np.double)
    state_pred[0,:] = state_prev[0,:] + 0.1 * v * np.cos(state_prev[2,:])
    state_pred[1,:] = state_prev[1,:] + 0.1 * v * np.sin(state_prev[2,:])
    state_pred[2,:] = circular_limit(state_prev[2,:] + 0.1 * w)
    return state_pred

def predicted_state(x_prev, P_prev, v, w):

	# calculate sigma points
	sigma_points = calculate_sigma_points(x_prev, P_prev, param)

	# propagate state
	x_prop = propagate_state(sigma_points,v,w)

	# calculate weights
	W_s, W_c = generate_weights(param)

	# predicted state
	x_pred = np.sum(W_s * x_prop,axis=1).reshape(5,1)

	# update covariance
	P_pred = np.zeros((5,5),dtype=np.double)

	for i in range(2*param['dims'] + 1):
		A = (x_prop[:,i] - x_pred[:,:])
		P = A @ A.T
		P = W_c[0,i] * P
		P_pred += P

#	B = (x_prop - x_pred)
#	P_pred = (W_c[0,:] * B) @ (B.T)

	print("P_pred: "+str(P_pred))
	return x_pred,P_pred

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
	z_pred[1,:] = circular_limit(phi)	

	return z_pred

def corrected_state(x_pred, P_pred, z, l):

	# calculate sigma points
	sigma_points = calculate_sigma_points(x_pred, P_pred, param)

	# calculate H
	z_prop = calculate_z_pred(sigma_points, l)

	# calculate weights
	W_s, W_c = generate_weights(param)

	# predicted state
	z_pred = np.sum(W_s * z_prop,axis=1).reshape(5,1)

	# update covariance
	P_pred = np.zeros((5,5),dtype=np.double)

	for i in range(2*param['dims'] + 1):
		A = (z_prop[:,i] - z_pred[:,:])
		P = A @ A.T
		P = W_c[0,i] * P
		P_pred += P

	# update kalman gain
	S = H @ P_pred @ H.T + V @ R @ V.T
	# kalman_gain = P_pred @ H.T @ pinv( H @ P_pred @ H.T + R)
	kalman_gain = P_pred @ H.T @ inv(S)

	x_est = x_pred + kalman_gain @ (z - z_pred)
	x_est[2,0] = circular_limit(x_est[2,0])
	# P_est = (1 - kalman_gain @ H) @ P_pred
	P_est = P_pred - kalman_gain @ S @ kalman_gain.T

	return x_est,P_est

def run_ekf(x_prev,P_prev,v,w,l,z):
	x_pred,P_pred = predicted_state(x_prev, P_prev, v, w)
	x_est, P_est  = corrected_state(x_pred, P_pred, z,l)
	return x_est,P_est

# init x and P
x_prev = np.array([x_true[0],y_true[0],th_true[0]],dtype=np.double)
P_prev = np.array([[1,0,0],[0,1,0],[0,0,0.1]],dtype=np.double)

z = np.zeros((2,1),dtype=np.double)

use_only_one_sensor = 0

if __name__ == '__main__':

	x = np.vstack( (x_prev.reshape(3,1),np.array([0,0]).reshape(2,1)))
	P = np.zeros((5,5))
	P[0:3,0:3] = P_prev
	P[3:5,3:5] = Q

	x_pred, P_pred = predicted_state(x,P,vel[1],om[1])

	P_pred[3:5,:] = 0
	P_pred[:,3:5] = 0
	P_pred[3:5,3:5] = R

	for i in range(0,len(l)):
		if r[1,i] != 0:
			z[0,0] = r[1,i]
			z[1,0] = b[1,i]
			landmark = l[i,:]
			break

	x_est,P_est = corrected_state(x_pred, P_pred, z, landmark)

	print(x_est)
	print(P_est)
	'''
	for j in range(1,len(x_true)): 
		# print("\niter: " + str(j))
		# x_prev = np.array([x_true[j],y_true[j],th_true[j]])

		v = vel[j]
		w = om[j]

		if use_only_one_sensor == 1:
			for i in range(0,len(l)):
				if r[j,i] != 0:
					z[0,0] = r[j,i]
					z[1,0] = b[j,i]
					landmark = l[i,:]
					break

			x_est,P_est=run_ekf(x_prev, P_prev, v, w, landmark, z)
		else:
			x_pred,P_pred = predicted_state(x_prev,P_prev,v,w)
			x_est = x_pred
			P_est = P_pred
			for i in range(0,len(l)):
				if r[j,i] != 0:
					z[0,0] = r[j,i]
					z[1,0] = b[j,i]
					landmark = l[i,:]					
					x_est, P_est = corrected_state(x_est, P_est, z, landmark)
			
		# print("x_est: ")
		# print(x_est)
		# print("P: ")
		# print(P_est)
		# print("x_true:")
		# print(x_true[j],y_true[j],th_true[j])
		x_prev = x_est
		P_prev = P_est

		error_x = x_est[0,0] - x_true[j]
		error_y = x_est[1,0] - y_true[j]
		error_theta = x_est[2,0] - th_true[j]

		if j%1000 == 0:
			print("\niter: " + str(j))
			print(x_est)
			print(P_est)
			print("x_error:" + str(error_x))
			print("y_error:" + str(error_y))
			print("theta_error:" + str(error_theta))
		
		if j==1:
			final_mu=x_est.T
			final_error_x = error_x
			final_error_y = error_y
			final_error_theta = error_theta
		else:
			final_mu = np.concatenate((final_mu,x_est.T))
			final_error_x = np.concatenate((final_error_x,error_x))
			final_error_y = np.concatenate((final_error_y,error_y))
			final_error_theta = np.concatenate((final_error_theta,error_theta))

	print("\naverage x error: " + str(np.sqrt(np.mean(final_error_x**2))))
	print("average y error: " + str(np.sqrt(np.mean(final_error_y**2))))
	print("average theta error: " + str(np.sqrt(np.mean(final_error_theta**2))))

	#ground thruth plot
	#plt.plot(l[:,0],l[:,1],'ro')
	#plt.plot(x_true, y_true, 'ro')
	#plt.show()

	#predictions 
	# plt.plot(l[:,0],l[:,1],'ro')
	# plt.plot(final_mu[:,0], final_mu[:,1],'bo')
	# plt.show()

	# plt.plot(final_error)
	# plt.show()
    '''