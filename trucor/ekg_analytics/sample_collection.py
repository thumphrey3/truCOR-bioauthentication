import os, sys
import pandas as pd
import time
from serial import Serial
import matplotlib.pyplot as plt 

values = []

def doAtExit():
	serialArduino = Serial("/dev/cu.usbmodemFA131", baudrate=115200)
	serialArduino.close()
	print("Connection to Arduino closed.")
	print("serialArduino.isOpen() = " + str(serialArduino.isOpen()))

def collect_EKG(device="Pulse Sensor", duration=10):
	if (device == "Pulse Sensor"):
		serialArduino = Serial("/dev/cu.usbmodemFA131", 9600)
		print("serialArduino.isOpen() = " + str(serialArduino.isOpen()))
		
		for i in range(0,200):
			values.append(0)

		t0 = time.time()
		t1 = time.time()
		while ((t1-t0) <= duration):
			while(serialArduino.inWaiting()==0):
				pass
			print("readline()")
			valueRead = serialArduino.readline(500)

			try:
				valueInInt = int(valueRead)
				print(valueInInt)
				if valueInInt <= 1024:
					if valueInInt >= 0:
						values.append(abs(valueInInt))
						t1=time.time()
					else:
						print("Invalid! Negative Number.")
				else:
					print("Invalid! Value too large.")

			except ValueError:
				print("Invalid! Cannot cast.")

		sps = int((len(values)-200)/20)
		no_samples = len(values)
		trace_data = pd.DataFrame(data=values, columns=['rawdata'])

	elif (device == "EKG Monitor"):
		serialArduino = Serial("/dev/cu.usbmodemFA131", baudrate=115200)
		print("serialArduino.isOpen() = " + str(serialArduino.isOpen()))
		
		for i in range(0,200):
			values.append(1000)

		t0 = time.time()
		t1 = time.time()
		values = []
		while ((t1-t0) <= duration):
			while(serialArduino.inWaiting()==0):
				pass
			print("readline()")
			valueRead = serialArduino.readline(500)

			try:
				valueInInt = int(valueRead)
				fmtValue = (valueInInt/1000)+5000
				if fmtValue <= 11000:
					if fmtValue >= -7000:
						values.append(fmtValue)
						t1=time.time()
					else:
						print("Invalid! Out of range value.")
				else:
					print("Invalid! Value too large.")

			except ValueError:
				print("Invalid! Cannot cast.")

		sps = int((len(values)-200)/duration)
		rawvalues = values[200:]
		# plt.plot(rawvalues, alpha=0.5, color='blue', label="raw signal")
		# plt.show()
		trace_data = pd.DataFrame(data=rawvalues, columns=['rawdata'])
		print(sps)
		print(trace_data)
	
	else:
		print('Device not supported.')

	return trace_data, sps



