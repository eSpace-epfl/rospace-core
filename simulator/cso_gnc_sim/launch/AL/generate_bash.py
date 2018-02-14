import numpy as np

steps=np.array([2.5,7.5,10,25,50,75,100,500,1000,2500,5000])

adA=np.concatenate([steps,[0],-steps])
adL=np.concatenate([steps,[0],-steps])

for a in adA:
	for l in adL:
		if a==0.0 and l==0.0:
			print ""	
		else:
			print "roslaunch testAL.launch adA:="+str(int(a))+" adL:="+str(int(l))

