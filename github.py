import numpy as np
import matplotlib.pyplot as plt     
from scipy import signal                  
from scipy.signal import find_peaks
from scipy.fft import fft, ifft           
import math                              
from scipy.io import wavfile
import time
class LOCALIZATION:
    soundspeed = 34120 #(cm/s)
    mic_locs = np.array([[0,0,50],[0,480,50],[480,480,50],[480,0,50],[0,240,80]])


    def __init__(self, sample_rate=44100, mic_locs=mic_locs):
        """
        Initiation

        Args:
            sample rate [Hz]
            locations of microphones [cm]
        Returns:
            an instance of the TDOA class
        """
        self.sample_rate = sample_rate
        self.period = 1/sample_rate

        self_mic_locs = mic_locs  

    # Load the reference signal from memory and take the first channel after 0.5s to aviod the noisy part in the beginning
        sample_rate, self.reference = wavfile.read("opnames/Reference.wav")
        self.reference = self.reference[int(0.5*self.sample_rate):int(1* self.sample_rate),0]
    
    def localization(self, recording, algorithm):
        """
        #calculates and returns the xyx coordinates of the estimated location
        #1 First a tdoa's list are calculated with tdoa_list method, based on the 5-channel that is given as input
        #2 either grid_xyz or matrix_xyz calculates the coordinates based on this tdoa list

        #paramters:
        #recording (numpy.ndarray) : a 5-channel recoring
        #algorithm (string)        : 'grid' or 'matrix' 

        #returns: 
        #location (list) :       xyz coordinates of the estimated location
        """

        tdoas = self.tdoa_list(self.reference, recording)
        print(tdoas)
        location = None
        if algorithm == 'grid':
            location = self.grid_xyz(tdoas)
            return location
        elif algorithm == 'matrix':
            location = self.matrix_xyz(tdoas) 
            return location
 #___________________TDOA_____________________________#    
    #Segmenting function
    @staticmethod
    def segment_audio(audio, segment_duration, sample_rate):
        """
        This funcion segments the recording into a short slice aroud where the peak sequence of the beacon are recorded.
        
        parameters :
        audio (numpyndarray): a 5-channel recording
        segment_duration (float) : duratoin of one segment
        sample rate () : sample rate of the audio

        returns:
        segment (numpyndarray) : a segment of the 5-channel recording

        """
        #1 the first channel is chosen and sliced starting from 0.5 seconds
        ch1 = audio[int(0.5*sample_rate):,0]

        # this is then normalized so max = 1 
        normal = ch1/np.max(abs(ch1))

        #2 Find all peaks above 0.8 in the entire audio signal after 0.5 seconds
        peaks, _ = find_peaks(normal, height=0.80)
        if len(peaks) == 0:
            raise ValueError("No peaks found in the audio signal.")
    
        # Get the first peak to determine the start of segment
        first_peak = peaks[0]

        #3 an offset is introduced and converted to number of samples and added to the index of the first peak
        offset_samples = int(0.25 * sample_rate)
        start_sample = first_peak + offset_samples

        #4 Segmenting the audio slightly after the first peak (after the offset)
        samples_per_segment = int(segment_duration * sample_rate)
        start_index =   1 * samples_per_segment + start_sample
        end_index = start_index + samples_per_segment

        segment = audio[start_index : end_index]
          
        return segment

    def tdoa_list(self, reference, recording):
        """
        This method calls tdoa function to calculate the tdoa for each pair of microphones
        and stores them in a list.

        parameters:
        reference (numpy.ndarray) : the best segment of the refernce signal (transmitted beacons signal)
        recording (numpy.ndarray) : a 5-channel recording
        
        returns:
        tdoa_list (list)   : tdoa list for the given recording (location)
        """
        x = reference
        y = np.array(recording)
        y_segment = self.segment_audio(y, 0.5, self.sample_rate)
 
        tdoa_list = []
        for i in range(5):
            for j in range(i + 1, 5):
                tdoa_list.append(self.tdoa(x, y_segment[:,i], y_segment[:,j]))

        print(tdoa_list)

        return tdoa_list

    def tdoa(self, ref, rec_ch1, rec_ch2):
        """
        this method calcaulate the tdoa for a microphones pairs (a pair of channels for one recording)

        parameters:
        ref (numpy.ndarray) : the best segment of the refernce signal (transmitted beacons signal)
        rec_ch1 (numpy.ndarray) : one channel of the recording
        rec_ch2 (numpy.ndarray) : another channel the same recording

        returns:
        tdoa (float) : a tdoa values for a microphones pair for the same recording
        """
        h1 = self.ch3(ref, rec_ch1, epsi=0.004)
        h2 = self.ch3(ref, rec_ch2, epsi=0.004)
        t_h1 = np.arange(0, len(h1) * self.period, self.period)
        t_h2 = np.arange(0, len(h2) * self.period, self.period)

        peak1_index = np.argmax(np.abs(h1))
        peak2_index = np.argmax(np.abs(h2))

        t_peak_1 = t_h1[peak1_index]
        t_peak_2 = t_h2[peak2_index]
        
        tdoa = t_peak_1 - t_peak_2

        return tdoa

    # Channel estimation
    @staticmethod
    def ch3(x,y,epsi):
        """
        This function estimates the channel's impulse response h[n]
        by taking the fft of y and x and pointwisely divide Y[k]/X[k] and taking the inverse fft.

        parameters:
        x (numpy.ndarray) : the reference signal 
        y (numpy.ndarray) : recording
        epsi (float)      : value around 0.001 to avoid division by zero, when X[k] = 0
        """ 
        Nx = len(x) # Length of x
        Ny = len(y) # Length of y
       
        Y = fft(y)
        X = fft(x)
        H = Y/(X + 1e-20)          

        # Threshold to avoid blow ups of noise during inversion
        ii = np.abs(X) < epsi*max(np.abs(X))

        for n in range(0, len(X)):
            if ii[n] == True:
              H[n] = 0

        h = np.real(ifft(H))   # ensure the result is real
        
        return h
    














#___________________LOCALIZATION_____________________________#    
    def TDOA_calc(self,xyz):
        """
        This function takes in xyz [cm] values and calculates the theoretical time differences of arrival between microphone pairs

        Args:
            xyz (np.array) locations [[x,y,z],[x,y,z], etc.]) [cm]
        Returns:
            allTDOA (np.array) an array of TDOA values [[10 TDOA values],[10 TDOA values], etc.]) [sec]
        """
        allTDOA = []
        mic_locs = np.array([[0,0,50],[0,480,50],[480,480,50],[480,0,50],[0,240,80]])
        for row in xyz:
            distances = []
            for loc in mic_locs:
                dist = np.linalg.norm(loc-row)
                distances.append(dist)
            times =  np.array(distances)/self.soundspeed
            TDOA = []
            for i in range(0,len(times)):
                for j in range(i+1,len(times)):
                    TDOA = np.append(TDOA,(times[i]-times[j]))
            allTDOA = np.concatenate((allTDOA,TDOA))
        allTDOA = np.reshape(allTDOA,(-1,10))
        return(allTDOA)
    #Get the Arest, B, Yrest, get the Acol0 matrix, do B-Arest*Yrest for getting [Acol0]*[x]=[B]-[Arest]*[Yrest]
    #Then use pinv to get [x] and return it.
    def matrix_xyz(self,TDOA):
        """
        matrix location estimator

        Args:
            TDOA (np.array) [10 TDOA values]
        Returns:
            part of Y vector (np.array) [x,y,z] cm approximate location
        """
        #get the difference vectors for column 0 of matrix A:
        #so one column of ten rows, each containing a row vector [x2-x1], [x3-x1], ...
        differences = np.array([])
        for i in range(0,len(self.mic_locs)):
            for j in range(i+1,len(self.mic_locs)):
                differences = np.append(differences,(self.mic_locs[j]-self.mic_locs[i]))
        #set A and B matrix
        row = 0
        Acol0 = 2*differences.reshape((10,3))
        Arest = np.zeros(40).reshape((10,4))
        B = np.array([])
        for i in range (0,5):
            for j in range(i+1,5):
                #get the -2rij entries of the A-matrix and put them in the right place
                r = TDOA[row]*self.soundspeed
                Arest[row,j-1] = -2*r
                row += 1
                #get the B entries
                B = np.append(B,r**2-np.linalg.norm(self.mic_locs[i])**2+np.linalg.norm(self.mic_locs[j])**2)
        #calculate [Y]
        A = np.concatenate((Acol0,Arest),axis=1)
        B = B.reshape((10,1))
        # print(np.round(A,2))
        Y = np.matmul(np.linalg.pinv(A),B)
        return(Y[0:3].reshape((3)))
    
    def grid_xyz(self,tdoa_real,size=10,begx=0,endx=480,begy=0,endy=480,z=30,iteration=0,cont_count=4): 
        """
        Grid search algorithm. Takes in a TDOA values estimates the location.

        Args:
            tdoa_real:              (np array) [10 entries] of TDOA values
            size:                   (int) this number decides how many points (n*n) the grid has per iteration
            begx, endx, begy, endy: (int) (don't change) the upper and lower bounds of the grid,
                                          this gets updated automaticcaly per iteration
            z:                      (int) z location of the grid (not very relevant)
            iteration:              (int) (don't change) the current iteration of the function. 
            cont_count:             (int) how many iterations are used. more=more accurate but diminishing returns and it takes more time.
        Returns:
            best (np array) [x,y,z] cm approximate location
        """
        #(gets nxn grid of coordinates evenly spaced) at height z
        #make grid
        xgrid = np.tile(np.linspace(begx,endx,size+2)[1:-1],size)
        ygrid = np.repeat(np.linspace(begy,endy,size+2)[1:-1],size)
        zgrid = np.repeat(z,size**2)
        grid = np.stack((xgrid,ygrid,zgrid),axis=1)
        #get the TDOA for each point in this grid
        gridTDOA = self.TDOA_calc(grid)
        #get the squared error for each point
        errors = np.array([])
        for row in gridTDOA:
            error = np.linalg.norm(row-tdoa_real)
            errors = np.append(errors,error)
        best = grid[np.argmin(errors)]
        #continue with the best point
        if iteration<cont_count:
            padding = 480/(size**(iteration+1))/2
            begx = best[0] - padding
            endx = best[0] + padding
            begy = best[1] - padding
            endy = best[1] + padding
            
            iteration += 1
            best = self.grid_xyz(tdoa_real,size,begx,endx,begy,endy,z,iteration,cont_count)
            return(best)
        else: return(best)



start_time = time.time()   
LOBI = LOCALIZATION()
sample_rate, rec = wavfile.read("opnames/record_x64_y40.wav")
print(type(rec))
loc = LOBI.localization(rec, algorithm='grid')
end_time= time.time()
print("time:", end_time-start_time)

print("fout",np.linalg.norm(loc[0:2]-[64,40]))
print(loc)