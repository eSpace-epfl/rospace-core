import numpy as np

steps=np.array([500,1000,5000,10000])
dt_steps = np.array([1,60,120,240])

adA=np.concatenate([steps,[0],-steps])
adL=np.concatenate([steps,[0],-steps])


for l in adL:
	for t in dt_steps:
		if l==0.0:
			print ""	
		else:
			print "roslaunch testALT.launch adA:=0 adL:="+str(int(l))+" dT:="+str(int(t))

