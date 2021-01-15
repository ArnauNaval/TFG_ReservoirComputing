import scipy.io
import numpy as np
import scipy.signal as spsg

class Data():
    def __init__(self,training_percentage):
        self.data = None
        self.training_data = None
        self.training_results = None
        self.test_data = None
        self.test_results = None
        self.spectral_bands = {
            'theta' : [4.,7.],
            'alpha' : [8.,15.],
            'beta'  : [15.,30.],
            'low gamma' : [30.,70.],
            'high gamma': [70.,100.],
            'low ripple': [100.,150.],
            'high ripple': [150.,200.],
            'low multi-unit': [200.,500.],
            'high multi-unit': [500.,1000.],
            'baseline': [4.,500.]
        }
        self.training_percentage = training_percentage
        self.test_percentage = 100-training_percentage
        self.fs = 2048
        self.num_columns = None
        self.num_trials_train = None 
        self.num_trials_test = None
        self.train_labels = None
        self.test_labels = None
        
    def import_data(self,file):
        """
            Loading data from MatLab, rounding to 4 decimals
        """
        self.data = scipy.io.loadmat(file)
        self.data = self.data['out'] 
        self.data = self.data.round(decimals=4)
        self.num_columns = self.data.shape[1]
        self.num_trials_train = int((self.data.shape[2]*self.training_percentage)/100)
        self.num_trials_test = self.data.shape[2]-self.num_trials_train
        print(f'Total of {self.data.shape}' )  
        
    def build_training_matrix(self):
        """
            
            Training matrix will have the shape [128,508*n_trials*states]
            
            order='F' means that data will be read/write with Fortran-like index order, due to data coming from MatLab
            
        """
        
        training_amount = self.num_trials_train
        #print(f'Amount of trials for training: {training_amount}')
        
        self.training_data = np.zeros([self.data.shape[0],self.data.shape[1]*training_amount*self.data.shape[3]])
        
        size = self.data.shape[1]*training_amount
        
        self.training_data[:,:size] = self.data[:,:,:training_amount,0].reshape(-1,size,order='F')
        self.training_data[:,size:size*2] = self.data[:,:,:training_amount,1].reshape(-1,size,order='F')
        self.training_data[:,size*2:size*3] = self.data[:,:,:training_amount,2].reshape(-1,size,order='F')
        self.training_data[:,size*3:size*4] = self.data[:,:,:training_amount,3].reshape(-1,size,order='F')
        self.training_data[:,size*4:] = self.data[:,:,:training_amount,4].reshape(-1,size,order='F')

        #print(f'Training data shape {self.training_data.shape}')

    def build_training_results(self):
        """
            
            Results matrix will have the shape [5,508*n_trials*states]
            
            order='F' means that data will be read/write with Fortran-like index order, due to data coming from MatLab
            
        """
        if self.training_data is None:
            print("You first need to build your training matrix")
            return 1
        
        training_amount = self.num_trials_train
                
        size = self.data.shape[1]*training_amount

        self.training_results = np.zeros((self.data.shape[-1],self.training_data.shape[1]), dtype=float)
        
        self.training_results[0,:size] = 0.9
        self.training_results[1,size:size*2] = 0.9
        self.training_results[2,size*2:size*3] = 0.9
        self.training_results[3,size*3:size*4] = 0.9
        self.training_results[4,size*4:] = 0.9
        
        #print(f'Training results shape {self.training_results.shape}')

        
    def build_test_matrix(self):
        """
            Training matrix will have the shape [128,508*y_trials*states]
            
            order='F' means that data will be read/write with Fortran-like index order, due to data coming from MatLab
            
        """
        
        test_amount = self.num_trials_test
        #print(f'Amount of trials for testing: {test_amount}')
        
        self.test_data = np.zeros([self.data.shape[0],self.data.shape[1]*test_amount*self.data.shape[3]])
        
        size = self.data.shape[1]*test_amount
        
        self.test_data[:,:size] = self.data[:,:,:test_amount,0].reshape(-1,size,order='F')
        self.test_data[:,size:size*2] = self.data[:,:,:test_amount,1].reshape(-1,size,order='F')
        self.test_data[:,size*2:size*3] = self.data[:,:,:test_amount,2].reshape(-1,size,order='F')
        self.test_data[:,size*3:size*4] = self.data[:,:,:test_amount,3].reshape(-1,size,order='F')
        self.test_data[:,size*4:] = self.data[:,:,:test_amount,4].reshape(-1,size,order='F')

        #print(f'Test data shape {self.test_data.shape}')

    def build_test_results(self):
        """
            
            Results matrix will have the shape [5,508*y_trials*states]
            
            order='F' means that data will be read/write with Fortran-like index order, due to data coming from MatLab
            
        """
        if self.test_data is None:
            print("You first need to build your test matrix")
            return 1
        
        test_amount = self.num_trials_test
                
        size = self.data.shape[1]*test_amount

        self.test_results = np.zeros((self.data.shape[-1],self.test_data.shape[1]), dtype=float)
        
        self.test_results[0,:size] = 0.9
        self.test_results[1,size:size*2] = 0.9
        self.test_results[2,size*2:size*3] = 0.9
        self.test_results[3,size*3:size*4] = 0.9
        self.test_results[4,size*4:] = 0.9
        
        #print(f'Testing results shape {self.test_results.shape}')

        
    def shuffle_data(self,data,results):
        data_shuffle = np.vstack((data,results))
        np.random.shuffle(data_shuffle.T)
        
        data = data_shuffle[:128,:]
        
        results = data_shuffle[128:,:]
        
        print(f'Training data shape after shuffle {data.shape}')
        print(f'Training results shape after shuffle {results.shape}')
        return data,results
    
    def filter_data(self,data,range_filter):
        low_freq, high_freq = self.spectral_bands[range_filter]
        #print(f'Band-pass filter between {low_freq} - {high_freq} Hz')
        
        low_freq, high_freq = low_freq/self.fs, high_freq/self.fs
        
        b,a = spsg.iirfilter(3, [low_freq,high_freq], btype='bandpass', ftype='butter')
        data = spsg.filtfilt(b, a, data, axis=1)
        
        return data

    def accuracy_lin(self,prediction,truth):
        acc = 0
        prediction = prediction.T
        #print(prediction.shape)
        #print(truth.shape)

        for i in range(prediction.shape[0]):
            tmp = [prediction[i,0],prediction[i,1],prediction[i,2],prediction[i,3],prediction[i,4]]

            res = tmp.index(max(tmp))


            tmp2 = [truth[i,0],truth[i,1],truth[i,2],truth[i,3],truth[i,4]]
            res2 = tmp2.index(max(tmp2))

            if res == res2:
                acc += 1

        return acc/prediction.shape[0]

    def build_train_labels_lin(self):
        self.train_labels = np.zeros([self.num_trials_train*5,5]) # 5 == Number of states
        self.train_labels[:self.num_trials_train,0] = 0.9
        self.train_labels[self.num_trials_train:self.num_trials_train*2,1] = 0.9
        self.train_labels[self.num_trials_train*2:self.num_trials_train*3,2] = 0.9
        self.train_labels[self.num_trials_train*3:self.num_trials_train*4,3] = 0.9
        self.train_labels[self.num_trials_train*4:,4] = 0.9
        
    def build_test_labels_lin(self):
        self.test_labels = np.zeros([self.num_trials_test*5,5]) # 5 == Number of states
        self.test_labels[:self.num_trials_test,0] = 0.9
        self.test_labels[self.num_trials_test:self.num_trials_test*2,1] = 0.9
        self.test_labels[self.num_trials_test*2:self.num_trials_test*3,2] = 0.9
        self.test_labels[self.num_trials_test*3:self.num_trials_test*4,3] = 0.9
        self.test_labels[self.num_trials_test*4:,4] = 0.9
    
    
    def process_data(self):
        #(128, 508, 140, 5)  ---->  (128,140,5)
        self.data = np.abs(self.data).mean(axis=1)
        
    def build_train_labels_log(self):
        self.train_labels = np.zeros([self.num_trials_train*5,]) # 5 == Number of states
        self.train_labels[:self.num_trials_train] = 0
        self.train_labels[self.num_trials_train:self.num_trials_train*2] = 1
        self.train_labels[self.num_trials_train*2:self.num_trials_train*3] = 2
        self.train_labels[self.num_trials_train*3:self.num_trials_train*4] = 3
        self.train_labels[self.num_trials_train*4:] = 4
        
    def build_test_labels_log(self):
        self.test_labels = np.zeros([self.num_trials_test*5,]) # 5 == Number of states
        self.test_labels[:self.num_trials_test] = 0
        self.test_labels[self.num_trials_test:self.num_trials_test*2] = 1
        self.test_labels[self.num_trials_test*2:self.num_trials_test*3] = 2
        self.test_labels[self.num_trials_test*3:self.num_trials_test*4] = 3
        self.test_labels[self.num_trials_test*4:] = 4

    def leftvsright_mixed(self):
        training_amount = self.num_trials_train
        
        self.training_data = self.data[:,:,:training_amount,0]
        self.training_data = np.hstack((self.training_data,self.data[:,:,:training_amount,5],
                                        self.data[:,:,:training_amount,1],self.data[:,:,:training_amount,6],
                                        self.data[:,:,:training_amount,2],
                                        self.data[:,:,:training_amount,7],
                                        self.data[:,:,:training_amount,3],
                                        self.data[:,:,:training_amount,8],
                                        self.data[:,:,:training_amount,4],
                                        self.data[:,:,:training_amount,9]
                                       ))

        self.training_data = self.training_data.reshape(self.training_data.shape[0],
                                                        self.training_data.shape[1]*self.training_data.shape[2], order='F')
        
        self.train_labels = np.zeros((training_amount*10))
        for col in range(0,self.train_labels.shape[0],10):
            self.train_labels[col] = 1
            self.train_labels[col+2] = 1
            self.train_labels[col+4] = 1
            self.train_labels[col+6] = 1
            self.train_labels[col+8] = 1
            
        

        test_amount = int((self.data.shape[2]*self.training_percentage)/100)
        
        self.test_data = self.data[:,:,test_amount:,0]
        self.test_data = np.hstack((self.test_data,self.data[:,:,test_amount:,5],
                                        self.data[:,:,test_amount:,1],self.data[:,:,test_amount:,6],
                                        self.data[:,:,test_amount:,2],
                                        self.data[:,:,test_amount:,7],
                                        self.data[:,:,test_amount:,3],
                                        self.data[:,:,test_amount:,8],
                                        self.data[:,:,test_amount:,4],
                                        self.data[:,:,test_amount:,9]
                                       ))

        self.test_data = self.test_data.reshape(self.test_data.shape[0],
                                                        self.test_data.shape[1]*self.test_data.shape[2], order='F')
        
        self.test_labels = np.zeros((self.num_trials_test*10))
        for col in range(0,self.test_labels.shape[0],10):
            self.test_labels[col] = 1
            self.test_labels[col+2] = 1
            self.test_labels[col+4] = 1
            self.test_labels[col+6] = 1
            self.test_labels[col+8] = 1
