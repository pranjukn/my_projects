import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time
import random
import sys
import warnings
import multiprocessing as mp
from multiprocessing import Process, current_process
import os

tic = time.time()

def create_koef_matrix(length, width,k,kr):
	N_dots = length*width
	A = np.zeros((N_dots, N_dots))
	# Diagonal_elements	
	for i in range(1,N_dots,1):
		A[i][i] = -4*k -kr			
	#sides
	#top_horizontal
	for i in range(1,length,1):
		A[i][i] = -3*k -kr
		#bottom_horizontal
		A[i+(width-1)*length][i+(width-1)*length] = -3*k -kr
	#left_vertical
	for i in range(length,N_dots,length):
		A[i][i] = -3*k -kr
		#right_vertical
		A[i-1][i-1] = -3*k -kr
	
	#corners
	A[0][0] = -k -k -kr
	A[length-1][length-1] = -k -k -kr
	A[N_dots-length][N_dots-length] = -k -k -kr
	A[N_dots-1][N_dots-1] = -k -k -kr

	#right_neighbours
	for i in range(0,N_dots-1,1):
		if((i+2) % length !=1):
			A[i][i+1] = k
		#bottom_neighbours
		if((i+length) < N_dots ):
			A[i][i+length] = k
		
	#left_neighbours
	for i in range(0,N_dots,1):
		if(i % length !=0):
			A[i][i-1] = k							
		#top_neighbours
		if((i-length+1) >0):
			A[i][i-length] = k

	return A
	
def method_with_traps(x,t,completed_matrix):
	dpvdt = np.matmul(completed_matrix,x)
	return dpvdt
				
def switch_method(x,k,kr,k_back,k_trap_r):
	switcher={
		k        :  k,
		-2*k - kr: -2*k_back - k_trap_r,
		-3*k - kr: -3*k_back - k_trap_r,
		-4*k - kr: -4*k_back - k_trap_r
	}
	return switcher.get(x,0)
	
def inserting_traps(initial_coef_matrix, traps_location_matrix,k,kr,k_back,k_trap_r, N_dots):
	C = initial_coef_matrix.copy()
	if(isinstance(traps_location_matrix, int)):
		return initial_coef_matrix
	else:	
		for i in range(0,N_dots):
			for j in range(0,N_dots):
				if(i in traps_location_matrix):
					C[i][j] = switch_method(C[i][j],k,kr,k_back,k_trap_r)		
		return C

def adjust_bonds_with_traps(A, Traps,k_back,N_dots):
	for i in range(0,N_dots):
		for j in range(0,N_dots):
			if(j in Traps and A[i][j] != 0 and j!=i):
				A[int(i)][j] = k_back
								
	return A

def adjust_bonds_between_traps(A, Traps, k_back, k_between_traps,N_dots):
	for i in Traps:
		for j in range(0,N_dots):
			if(j in Traps and A[int(i)][j] != 0 and j != i):
				A[int(i)][j] = k_between_traps
				A[int(i)][int(i)] = A[int(i)][int(i)] + k_back - k_between_traps
								
	return A
	
def exponent1_fit(x,a, b):
	return a*np.exp(x*-b)
		
def exponent2_fit(x, a, b, c, d):
    return a * np.exp(b * x) + c * np.exp(d * x)
	
def exponent3_fit(x, a, b, c, d, e, f):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e * np.exp(-f * x)

def perform_fitting(A,t):	
	try:
		popt, pcov = curve_fit(exponent3_fit, t, A)
		#popt, pcov = curve_fit(exponent3_fit, t, A, method = 'trf')
	except RuntimeError:
		print("Runtime exception found")
		popt, pcov = np.zeros((6,)),np.zeros((6,6))
		pass
	return popt, pcov
	
def generating_number_of_traps(N_red_average):
	p = N_red_average/N_dots
	A = np.zeros((N_realizations))
	for j in range(0,N_realizations):
		A[j] = int(np.random.binomial(N_dots, p, 1))
		if (A[j] > N_dots):
			A[j] = N_dots
		if (A[j] < 0):
			A[j] = 0
	return A

def run_through_realizations(mixed_array,traps_matrix,k,kr,k_back,k_between_traps,k_trap_r,sample_number,return_dict,length,width,N_realizations,run_time,steps_per_time):
	N_dots = length * width
	
	time_steps = run_time * steps_per_time
	
	traps_index = np.linspace(0,N_dots-1,N_dots)
		
	t = np.linspace(0,run_time,time_steps)
	x0 = np.zeros(N_dots)
	x0[:] = 1/N_dots

	x = np.zeros(N_dots)
	
	red_sum = np.zeros((time_steps))
	initial_matrix = create_koef_matrix(length,width,k,kr)
	
	for real_nr in range(0,N_realizations):	
		# Generates RED indexes
		traps_location_matrix = random.sample(traps_index.tolist(), k=int(traps_matrix[real_nr]))
		#print("numeris sample " + str(l) + " numeris N realizations: " + str(real_nr) + str(traps_location_matrix))
		matrix_with_traps = inserting_traps(initial_matrix,traps_location_matrix,k,kr,k_back,k_trap_r,N_dots)

		matrix_with_traps = adjust_bonds_with_traps(matrix_with_traps, traps_location_matrix,k_back,N_dots)
		#print(matrix_with_traps)

		matrix_with_traps = adjust_bonds_between_traps(matrix_with_traps, traps_location_matrix,k_back, k_between_traps,N_dots)
		# Final_matrix
		#print(matrix_with_traps)
		
		x = odeint(method_with_traps,x0,t,(matrix_with_traps,))
		mixed_array += np.sum(x,axis = 1)
		for i in range(0,N_dots):
			if(i in traps_location_matrix):
				red_sum += x[:,i]
	
	green_sum = mixed_array - red_sum
	
	mixed_array= mixed_array/N_realizations	
	red_averaged = red_sum/N_realizations
	green_averaged = green_sum/N_realizations
	#fitting
	popt, pcov = perform_fitting(green_averaged,t)
	
	process_name = current_process().name
	print(f"Process ID: {process_name}")
	#print(popt)
	return_dict[sample_number] = popt
	
	# print("Coefficients of exponential sum from a to d accordingly:")
	# plt.figure(sample_number)
	# plt.title("Plot of averages")
	# plt.plot(t,mixed_array ,'b-', label = "sum_avg")
	# plt.plot(t,green_averaged,'g-', label = "green_avg")
	# plt.plot(t,red_averaged,'r-', label = "red_avg")
	# plt.plot(t, exponent3_fit(t,*popt), 'y-', label='Fitting_green')
	# # plt.plot(t, exponent3_fit(t,*return_dict[sample_number]), 'k.', label='return_dict')
	# plt.grid(True)
	# plt.legend(loc='best')
	# plt.show()


if __name__ == "__main__":	
	
	#setting_initial_values	
	number_of_samples = 70
	
	length = 5
	width = 5
	N_dots = length * width
	
	run_time = 15000
	steps_per_time = 1
	time_steps = run_time * steps_per_time
	
	N_realizations = 10

	# No need for seperate time constants for jumps between red states. So k_between_red = k_hop.
	#GREEN
	k_hop = 1/np.random.randint(25,high = 201, size = number_of_samples)
	k_dis = 1/np.random.randint(5000,high = 5001, size = number_of_samples)
	#RED
	k_RG = 1/np.random.randint(25,high = 30001, size = number_of_samples)
	k_between_red = k_hop
	k_trap = 1/np.random.randint(50,high = 5001, size = number_of_samples)

	eksponentes = np.zeros((number_of_samples,7))
		
	mixed_array = np.zeros((time_steps))
	empty_array = np.zeros((number_of_samples,time_steps))
	traps_matrix = np.zeros((number_of_samples,N_realizations))
	red_averaged = np.zeros((time_steps))

	return_dict = mp.Manager().dict()
	processes = []
	pool = mp.Pool(mp.cpu_count())
	for i in range(0,number_of_samples):
		print("Starting process number " + str(i))
		eksponentes[i][0] = random.uniform(0.5, 15)
		traps_matrix[i] = generating_number_of_traps(eksponentes[i][0])
		pool.apply_async(run_through_realizations, args = (mixed_array,traps_matrix[i],k_hop[i],
															k_dis[i],k_RG[i],k_hop[i],k_trap[i],i,return_dict,length,width,
															N_realizations,run_time,steps_per_time))	
	pool.close()
	pool.join()
	
	file1write=open("sample_file_mp.txt",'w')

	for i in range(0,number_of_samples):
		file1write.write(str(k_hop[i]) + " " + str(k_dis[i]) + " " + str(k_RG[i])   + " " + str(k_trap[i]) 
		+ " " +  str(eksponentes[i][0])
		+ " " +  str(return_dict[i][0])+ " " +  str(-return_dict[i][1])
		+ " " +  str(return_dict[i][2]) + " " +  str(-return_dict[i][3])
		+ " " +  str(return_dict[i][4])+ " " +  str(-return_dict[i][5])			
		+ '\n')
		
	file1write.close()
	
	toc = time.time()
	print("toc-tic: " +str(toc-tic))
	
