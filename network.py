import numpy as np
import scipy.linalg
from sklearn.linear_model import LinearRegression,LogisticRegression
import sklearn.neighbors as sklnn

class Network():
    def __init__(self, T = None, n_min = None, K = None, N = None, L = None, W_in = None, W = None, W_back = None, W_out = None): 
        
        self.T = T #number of training time steps (integer)
        self.n_min = n_min #time steps dismissed (integer)
        
        self.K = K #dimension of the input (integer) (may be None)
        self.N = N #dimension of the reservoir, i.e, number of nodes (integer)
        self.L = L #dimension of the output (integer)
        
        self.W_in = W_in #input connections (matrix of size self.N x self.K)
        self.W = W #adjacency matrix (matrix of size self.N x self.N)
        
  
        self.initial_state = None #initial state of the reservoir (state forgetting property)        
        self.trajectories = None #dynamics of the reservoir (matrix of size self.T x self.N) 
        self.regressor = None #regressor
        self.y_teach = None #desired output of the network (matrix of size self.L x self.T)
        self.y_teach_test = None #y_teach for doing the test (matrix of size self.L x (t_dismiss+t_autonom))                
        self.u = None #input (matrix of size self.K x self.T) 
        self.u_test = None #input durint training (matrix of size self.K x t_dismiss+t_autonom) 
        
        self.mean_train_matrix = None #matrix where we will perform the mean of each state in training
        self.mean_test_matrix = None #matrix where we will perform the mean of each state in testing
        
    def setup_network(self,d,k,inpu,reser,states):

        self.u = d.training_data

        self.mean_train_matrix = np.zeros([k,d.num_trials_train,states])


        ########################
        # Input Layer
        ########################

        self.W_in = np.zeros([self.N,self.K]) #input matrix

        for i in np.arange(self.N):
            for j in np.arange(self.K):
                p = np.random.uniform()
                if 0 <= p <= inpu:
                    self.W_in[i,j] = np.random.uniform(-1., 1.)            

                else:
                    self.W_in[i,j] = 0


        ########################
        # Reservoir Layer
        ########################

        self.W = np.zeros([self.N,self.N]) #adjacency matrix

        for i in np.arange(self.N):
            for j in np.arange(self.N):
                p = np.random.uniform()
                if 0 <= p <= reser:
                    self.W[i,j] = np.random.normal(loc=0, scale=1, size=1)[0]         

                else:
                    self.W[i,j] = 0


		########################
        # Making sure the largest eigenvalue in module is < 1
        ########################
        alpha = 0.22/max(abs(scipy.linalg.eigvals(self.W)))
        self.W = alpha*self.W

    
    def compute_nodes_trajectories(self,num_columns, num_trials, test=False, t_autonom=None): 
        """
        If test=False:
            -It computes self.trajectories, which is a matrix of size TxN, where each column is the trajectory of 
            a node. Notice that the first row corresponds to the initial state
        
        If test=True
            -Computes the predictions for the desired t_autonom
        """
        
        #initial state of the reservoir
        if test == False:
            x_prev = self.initial_state  
        if test == True:
            x_prev = self.trajectories[-1,:]
            

        if test == False:
            columna = -1
            trial = -1
            
            self.trajectories = np.zeros((self.T,self.N))
            
            for n in np.arange(self.T):
                x = np.tanh(np.dot(self.W_in,self.u[:,n])+np.dot(self.W,x_prev)) #state update equation
                self.trajectories[n,:] = x
                x_prev = x    
                
                if n%num_columns == 0: #mean for each state in a trial
                    self.mean_train_matrix[:,columna,trial] = self.mean_train_matrix[:,columna,trial]/num_columns
                    columna += 1

                if n %(num_columns*num_trials) == 0: #after each trial, we move on to the next one
                    trial += 1
                    columna = 0
                    
                self.mean_train_matrix[:,columna,trial] += x
                    
            return self

        elif test == True: 
            columna = -1
            trial = -1
            
            for n in np.arange(t_autonom):
                x = np.tanh(np.dot(self.W_in, self.u_test[:,n])+np.dot(self.W,x_prev)) #state update equation
                x_prev = x
                
                if n%num_columns == 0:#mean for each state in a trial
                    self.mean_test_matrix[:,columna,trial] = self.mean_test_matrix[:,columna,trial]/num_columns
                    columna += 1

                if n%(num_columns*num_trials) == 0:#after each trial, we move on to the next one
                    trial += 1
                    columna = 0   

                self.mean_test_matrix[:,columna,trial] += x

            return self
            
             
    def train_network(self, num_states, classifier ,num_columns, num_trials, labels, num_nodes):
        """
			Method responsible for processing the training data trough the network and fitting the result to the desired classifier
		"""

		#Define the initial state (which is not relevant due to state forgetting property)
        self.initial_state = np.ones(self.N)

		#Data trough network
        self.compute_nodes_trajectories(num_columns, num_trials)

        self.mean_train_matrix = self.mean_train_matrix.reshape((num_nodes,num_trials*num_states),order='F')

        if classifier == 'lin':
            regressor = LinearRegression()
            regressor.fit(self.mean_train_matrix.T, labels)

        elif classifier == 'log':
            regressor = LogisticRegression(max_iter=100000)
            regressor.fit(self.mean_train_matrix.T, labels.T)

        elif classifier == '1nn':
            regressor = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')
            regressor.fit(self.mean_train_matrix.T, labels.T)

        self.regressor = regressor

        return self  
    
    def test_network(self, data, num_columns, num_trials, num_nodes, states, t_autonom):
        """
    		Method responsible for processing the testing data trough the network
        """ 
        
        self.u_test = data                        
                      
        self.compute_nodes_trajectories(num_columns, num_trials, test=True, t_autonom=t_autonom)
        
        self.mean_test_matrix = self.mean_test_matrix.reshape((num_nodes,num_trials*states),order='F')

        return self
            