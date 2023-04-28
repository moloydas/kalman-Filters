import numpy as np
import math
from numpy.linalg import inv,pinv
import matplotlib.pyplot as plt

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
b_var=dataset['b_var'][0][0]
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

Q = np.array([ [v_var,0], [0,om_var] ],dtype='float')
R = np.array([ [r_var,0], [0,b_var] ], dtype='float')
V = np.eye(2, dtype='float')

def circular_limit(theta):
	if theta > math.pi:
		new_theta = -1 * (2*math.pi - theta)
	elif theta < -math.pi:
		new_theta = -1 * (-2*math.pi - theta)
	else:
		new_theta = theta
	return new_theta

def calculate_H_and_z_pred(x_pred, l):
	H = np.zeros((2,3),dtype='float')
	z_pred = np.zeros((2,1),dtype='float')

	# measured range and angle of landmark
	a = l[0] - x_pred[0,0] - d*np.cos(x_pred[2,0])
	b = l[1] - x_pred[1,0] - d*np.sin(x_pred[2,0])
	
	r = np.sqrt( np.square(a)+np.square(b) )
	phi = math.atan2(b, a) - x_pred[2,0]
	z_pred[0,0] = r
	z_pred[1,0] = circular_limit(phi)	

	# H first row
	H[0,0] = -a/np.sqrt(np.square(a)+np.square(b))
	H[0,1] = -b/np.sqrt((np.square(a)+np.square(b)))
	H[0,2] = (-d*b*np.cos(x_pred[2,0])+d*a*np.sin(x_pred[2,0]))/np.sqrt((np.square(a)+np.square(b)))

	# H second row
	H[1,0] = b/(np.square(a)+np.square(b))
	H[1,1] = -a/(np.square(a)+np.square(b))
	H[1,2] = ( (-d*b*np.sin(x_pred[2,0])-d*a*np.cos(x_pred[2,0])) / (np.square(a)+np.square(b))) - 1
	return H,z_pred

def calculate_F(x_prev,v,w):
	F = np.eye(3,dtype='float')
	F[0,2] = -1 * np.sin(x_prev[2,0]) * v * 0.1
	F[1,2] = np.cos(x_prev[2,0]) * v * 0.1
	return F

def calculate_W(x_prev):
	W = np.zeros((3,2),dtype='float')
	W[0,0] = 0.1 * np.cos(x_prev[2,0])
	W[1,0] = 0.1 * np.sin(x_prev[2,0])
	W[2,1] = 0.1
	return W

def predicted_state(x_prev, P_prev, v, w):
	x_pred = np.zeros((3,1),dtype='float')

	# propagate state
	x_pred[0,0] = x_prev[0,0] + 0.1 * v * np.cos(x_prev[2,0])
	x_pred[1,0] = x_prev[1,0] + 0.1 * v * np.sin(x_prev[2,0])
	x_pred[2,0] = circular_limit(x_prev[2,0] + 0.1 * w)

	# calculate F
	F = calculate_F(x_prev,v,w)

	# calculate W
	W = calculate_W(x_prev)

	# update covariance
	P_pred = F @ P_prev @ F.T + W @ Q @ W.T

	return x_pred,P_pred

def corrected_state(x_pred, P_pred, z, l):

	# calculate H
	H, z_pred = calculate_H_and_z_pred(x_pred, l)

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
x_prev = np.array([x_true[0],y_true[0],th_true[0]],dtype='float')
P_prev = np.array([[1,0,0],[0,1,0],[0,0,0.1]],dtype='float')

z = np.zeros((2,1))

use_only_one_sensor = 0

if __name__ == '__main__':

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

		if j%100 == 0:
			print("\niter: " + str(j))
			print("x_est:\n ")
			print(x_est)
			print("P_est:\n ")
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

		if j%1000 == 0:
			plt.plot(l[:,0],l[:,1],'ro')
			plt.plot(final_mu[:,0], final_mu[:,1],'bo')
			plt.pause(0.005)

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
	plt.show()

	# plt.plot(final_error)
	# plt.show()
