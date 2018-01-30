from datetime import datetime
import time 

import pandas as pd 
from scipy import stats
from scipy.signal import argrelextrema
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import numpy as np 
import matplotlib.pyplot as plt 
import math

import matplotlib.font_manager
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

import sqlite3 
from sqlite3 import Error


def personaldata():
	dataset = pd.read_csv("/Users/thumphrey/projects/trucor/th_log.csv")
	th_dataset = dataset.ECG[4000:8600]*(-1)
	return th_dataset

#Filter Functions
def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = .5*fs
	low = lowcut/nyq 
	high = highcut/nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def find_RPeaks(data):
	window = []
	RPeaks = []
	listpos = 200
	threshold = 3000
	for datapoint in data:
		if (datapoint <= threshold) and (len(window) <= 1):
			listpos +=1
		elif (datapoint > threshold):
			window.append(datapoint)
			listpos +=1
		else:
			maximum = max(window)
			beatposition = listpos - len(window) + (window.index(max(window)))
			RPeaks.append(beatposition)
			window=[]
			listpos +=1
	print(RPeaks)
	return RPeaks

def feature_extraction(data, peak_references, fs):
	S_Fid_List = []
	Q_Fid_List = []
	P_Fid_List = []
	T_Fid_List = []
	L_Fid_List = []
	TP_Fid_List = []
	L = []

	no_rows = len(peak_references)
	trace_container = np.zeros((no_rows, 7), dtype=int)

	#Heart rate should not be lower than 30BPM or higher than 130BPM
	hi_bpm = int(fs*(60/130))
	lo_bpm = int(fs*(60/30))

	i=0
	while(i < (len(peak_references)-1)):
		trace_container[i][3]= peak_references[i]
		trace_container[i+1][3]=peak_references[i+1]

		r0 = peak_references[i]
		r1 = peak_references[i+1]

		if (r1-r0 < hi_bpm) or (r1-r0 > lo_bpm):
			i +=1 
		else:
			s_window = int(.1*fs)
			s_end = r0+s_window

			q_window = int(.1*fs)
			q_start = r1-q_window


			p_window = int(.15*fs)

			S_Snap = data[r0:s_end]
			if (len(S_Snap) > 0):
				S_pos = S_Snap.idxmin()
				trace_container[i][4] = S_pos
				S_Fid_List.append(S_pos)

			Q_Snap = data[q_start:r1]

			if (len(Q_Snap) > 0):
				Q_pos = Q_Snap.idxmin()
				trace_container[i+1][2]= Q_pos
				Q_Fid_List.append(Q_pos)

			p_start = Q_pos-p_window
			p_end = Q_pos-1
			P_Snap = data[p_start:p_end]
			if (len(P_Snap) > 0):
				P_pos = P_Snap.idxmax()
				trace_container[i+1][1]= P_pos
				P_Fid_List.append(P_pos)

			l_prime_start = P_pos - int(.1*fs)
			l_prime_end = P_pos - 1
			L_prime_search = data[l_prime_start:l_prime_end]
			L = []
			l_pos = 0
			for pt in L_prime_search:
				X_x = l_prime_end
				X_y = data[X_x]
				Z_x = l_prime_start
				Z_y = data[Z_x]
				XY_x = (l_prime_start + l_pos) - l_prime_end
				XZ_x = l_prime_end - l_prime_start
				XY_y = pt-data[X_x]
				XZ_y = data[Z_x]-data[X_x]
				XY = [XY_x, XY_y]
				XZ = [XZ_x, XZ_y]
				cp = np.cross(XZ,XY)
				XZ_magnitude = math.sqrt((Z_x-X_x)**2 + (Z_y-X_y)**2)
				delta = abs(cp)/XZ_magnitude
				L.append(delta)
				l_pos += 1

			if (len(L) > 0):
				l_prime = l_prime_start + np.argmax(L)
				trace_container[i+1][0] = l_prime
				L_Fid_List.append(l_prime)


			t_start = S_pos + 1
			t_end = t_start + int(.2*fs)
			T_Snap = data[t_start:t_end]
			if (len(T_Snap) > 0):
				T_pos = T_Snap.idxmax()
				trace_container[i][5]=T_pos
				T_Fid_List.append(T_pos)

			t_prime_start = T_pos + int(.02*fs)
			t_prime_end = T_pos + int(.15*fs)
			t_prime_search = data[t_prime_start:t_prime_end]
			TP=[]
			t_prime_pos = 0
			for pt in t_prime_search:
				X_x = t_prime_start
				X_y = data[X_x]
				Z_x = t_prime_end
				Z_y = data[Z_x]
				XY_x = (t_prime_pos + t_prime_start)-X_x
				XZ_x = Z_x - X_x
				XY_y = pt-data[X_x]
				XZ_y = data[X_x]-data[Z_x]
				XY=[XY_x, XY_y]
				XZ=[XZ_x, XZ_y]
				cp = np.cross(XZ,XY)
				XZ_magnitude = math.sqrt((X_x-Z_x)**2 + (X_y-Z_y)**2)
				delta = abs(cp)/XZ_magnitude
				TP.append(delta)
				t_prime_pos +=1
			if (len(TP) > 0):
				t_prime = np.argmax(TP)+t_prime_start
				trace_container[i][6] = t_prime
				TP_Fid_List.append(t_prime)

			i += 1

	trace_df = pd.DataFrame(trace_container, columns=['L Prime', 'P', 'Q', 'R', 'S', 'T', 'T Prime'])
	print(trace_df)
	trace_df['RQ'] = (trace_df['R']-trace_df['Q'])/fs
	trace_df['RS'] = (trace_df['S']-trace_df['R'])/fs
	trace_df['RP']=(trace_df['R']-trace_df['P'])/(trace_df['T Prime']-trace_df['L Prime'])
	trace_df['RT']=(trace_df['T']-trace_df['R'])/(trace_df['T Prime']-trace_df['L Prime'])
	trace_df['ST']=(trace_df['T']-trace_df['S'])/(trace_df['T Prime']-trace_df['L Prime'])
	trace_df['PQ']=(trace_df['Q']-trace_df['P'])/(trace_df['T Prime']-trace_df['L Prime'])
	trace_df['PT']=(trace_df['T']-trace_df['P'])/(trace_df['T Prime']-trace_df['L Prime'])
	trace_df['RL']=(trace_df['R']-trace_df['L Prime'])/(trace_df['T Prime']-trace_df['L Prime'])
	trace_df['LQ']=(trace_df['Q']-trace_df['L Prime'])/(trace_df['T Prime']-trace_df['L Prime'])
	trace_df['STPrime']=(trace_df['T Prime']-trace_df['S'])/(trace_df['T Prime']-trace_df['L Prime'])
	trace_df['RTPrime']=(trace_df['T Prime']-trace_df['R'])/(trace_df['T Prime']-trace_df['L Prime'])
	trace_df = trace_df.replace([np.inf, -np.inf], np.nan)
	trace_df = trace_df.dropna(axis=0, how='any')
	return trace_df

def remove_trace_outliers(trace_df):
	percent_clean = 0
	z = .5
	while(percent_clean < .7):
		clean_trace_df = trace_df[(np.abs(stats.zscore(trace_df)) < z).all(axis=1)]
		raw_length = len(trace_df.axes[0])
		clean_length = len(clean_trace_df.axes[0])
		percent_clean = round((clean_length/raw_length), 1)
		z +=.01

	return clean_trace_df

def set_template(snippet, fs):
	hrw = 0.75
	lowcut = 2
	highcut = 40

	raw_df= pd.DataFrame(data=snippet, columns= ['rawdata'])
	mov_avg = raw_df.rawdata.rolling(int(hrw*fs)).mean()
	
	filtered_data = butter_bandpass_filter(snippet,lowcut,highcut,fs,order=5)

	raw_df = pd.DataFrame(data=snippet, columns= ['rawdata'])

	mov_avg = raw_df.rawdata.rolling(int(hrw*fs)).mean()
	avg_hr = (np.mean(raw_df.rawdata)) 
	mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
	mov_avg = [x*1.25 for x in mov_avg] 
	raw_df['data_rollingmean'] = mov_avg

	peaklist = find_RPeaks(raw_df['rawdata'])
	print('R Peaks detected and logged.')

	trace_frame = feature_extraction(raw_df.rawdata, peaklist, fs)
	print('Features extracted from ECG trace.')
	clean_trace_extended = remove_trace_outliers(trace_frame)
	print('Outliers removed from ECG template.')

	clean_trace_df = clean_trace_extended.drop(['L Prime','P','Q','R','S','T','T Prime'], axis=1)

	return clean_trace_df

def heart_identify(test_input, train_input, train_output):
	clf = LinearDiscriminantAnalysis()
	clf.fit(train_input, train_output)
	pred_test = clf.predict(test_input)
	user_id = stats.mode(pred_test, axis=None)[0][0]
	return user_id

def authenticate(test_input, positive_input, negative_input):
	positive_input['associated_account'] = 1
	plus_y_train = positive_input['associated_account']
	plus_X_train = positive_input.drop(['created_at', 'associated_account'], axis=1)
	negative_input['associated_account'] = -1
	neg_y_train = negative_input['associated_account']
	neg_X_train = negative_input.drop(['created_at', 'associated_account'], axis=1)

	X_restacked_frame = [plus_X_train, neg_X_train]
	x_re_frame = pd.concat(X_restacked_frame, ignore_index=True)
	X = x_re_frame.as_matrix()

	y_restacked_frame = [plus_y_train, neg_y_train]
	y_re_frame = pd.concat(y_restacked_frame, ignore_index=True)
	y_train = y_re_frame.as_matrix()

	scaler = StandardScaler()
	scaler.fit(X)
	X_train = scaler.transform(X)
	X_pred = scaler.transform(test_input)

	log_clf = LogisticRegression()
	log_clf.fit(X_train, y_train)
	pred_test = log_clf.predict(X_pred)
	pred_probability = log_clf.predict_proba(test_input)
	print(pred_probability)

	yes_count = sum(1 for x in pred_test if x > 0)
	no_count = sum(1 for x in pred_test if x < 0)
	total_count = len(pred_test)
	yes_percent = yes_count/total_count

	log_yes_percent = pred_probability.mean(0)

	verification = False 

	if (yes_percent >= .70):
		verification = True
		print('Yes \nUser Authenticated.')
	elif((yes_percent < .70) and (yes_percent >= .40)):
		print('Please attempt to re-authenticate.')
	else:
		print('No \nUser Not Authenticated.')

	return verification, yes_percent



