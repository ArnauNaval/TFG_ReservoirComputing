import sys
import numpy as np
import network as Network
import data as Data

#Retrieving arguments from the command line
test_name = str(sys.argv[1])
filter_name = str(sys.argv[2])
classifier = str(sys.argv[3])
num_nodes = int(sys.argv[4])
input_probability = float(sys.argv[5])
reservoir_probability = float(sys.argv[6])

d = Data.Data(80) #80% training 20% testing

Network = Network.Network()

#Setting the right data for all the possible combinations of problems and classifiers

if test_name == '5s':
	d.import_data('dataSorted_allOrientations.mat')
	if classifier == 'lin':
		d.build_train_labels_lin()
		d.build_test_labels_lin()
		
	elif classifier == 'log':
		d.build_train_labels_log()
		d.build_test_labels_log()

	else:
		print("This classifier is not supported for this test.")
		sys.exit(1)

	d.build_training_matrix()
	d.build_test_matrix()
	Network.L = 5

elif test_name == 'lvr':
	if classifier == 'log' or classifier == '1nn':
		d.import_data('dataSorted_leftAndRight.mat')
		d.leftvsright_mixed()
		Network.L = 1

	else: 
		print("This classifier is not supported for this test.")
		sys.exit(1)

else:
	print("This test does not exist.")
	sys.exit(1)

#Filtering the data
if filter_name not in d.spectral_bands.keys():
	print("The specified frequency band is not supported")
	sys.exit(1)

d.training_data = d.filter_data(d.training_data,filter_name)
d.test_data = d.filter_data(d.test_data,filter_name)


#Computing the absolute value of the data, to get rid of negative numbers
d.training_data = np.abs(d.training_data)
d.test_data = np.abs(d.test_data)

########################
# Define the network parameters
########################

Network.T = d.training_data.shape[1] #Number of training time steps
Network.n_min = 2540 #Number time steps dismissed
Network.K = 128 #Input layer size
Network.N = num_nodes #Reservoir layer size


Network.u = d.training_data
Network.y_teach = d.training_results

Network.setup_network(d,num_nodes,input_probability,reservoir_probability,d.data.shape[-1])

Network.train_network(d.data.shape[-1],classifier,d.num_columns, d.num_trials_train, d.train_labels, Network.N) 

Network.mean_test_matrix = np.zeros([Network.N,d.num_trials_test,d.data.shape[-1]])

Network.test_network(d.test_data, d.num_columns,d.num_trials_test, Network.N, d.data.shape[-1], t_autonom=d.test_data.shape[1])

if classifier == 'lin':
	print(f'Performance for {test_name} using {classifier} : {d.accuracy_lin(Network.regressor.predict(Network.mean_test_matrix.T),d.test_labels)}')

elif classifier == 'log':
	print(f'Performance for {test_name} using {classifier} : {Network.regressor.score(Network.mean_test_matrix.T,d.test_labels.T)}')

elif classifier == '1nn':
	print(f'Performance for {test_name} using {classifier} : {Network.regressor.score(Network.mean_test_matrix.T,d.test_labels)}')
