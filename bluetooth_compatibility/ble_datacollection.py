import os, sys
import pandas as pd
import time
import bluepy import btle
import math
import matplotlib.pyplot as plt 

#hex string to signed integer
def htosi(val):
	uintval = int(val, 16)
	bits = 4 * (len(val)-2)
	if uintval >= math.pow(2, bits-1):
		uintval = int(0 - (math.pow(2, bits) - uintval))
	return uintval

def collect_EKG(device="Bluetooth Connected Device", duration):
	if (device == "Bluetooth Connected Device"):
		ecg_uuid = btle.UUID(0x2A37)
		p = btle.Peripheral("C4:45:2F:37:D4:4C", "random")
		svc = p.getServiceByUUID(0x180D)
		ch = svc.getCharacteristics(ecg_uuid)[0]
		
		for i in range(0,200):
			values.append(1000)

		t0 = time.time()
		t1 = time.time()
		values = []
		while ((t1-t0) <= duration):
			print("readline()")
			valueRead = ch.read()

			try:
				valueInInt = valueRead[3:]
				pm_value = htosi(valueInInt)
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



# target_name = "TruCOR Bluefruit ECG Monitor"
# target_address = None

# nearby_devices = bluetooth.discover_devices()

# for bdaddr in nearby_devices:
# 	if target_name == bluetooth.lookup_name( bdaddr ):
# 		target_address = bdaddr
# 		break

# if target_address is not None:
# 	print("Found target bluetooth device with address ", target_address
# else: 
# 	print ("Could not find target bluetooth device nearby")