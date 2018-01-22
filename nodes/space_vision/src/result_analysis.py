#!/usr/bin/env python

""" This script is used for further analysis of the results files produced by space_vision.py

    Author: Gaetan Ramet
    License: TBD
"""

import os, sys, time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--measured_data_path', default=None)
    parser.add_argument('-r', '--real_data_path', default=None)

    args = vars(parser.parse_args())

    if args['measured_data_path'] is not None:
        save_file_path = os.path.join(args['measured_data_path'], 'measured_data.txt')

        with open(save_file_path, 'r') as save_file:
            data = save_file.readlines()

        time_m = np.asarray([])
        range_m = np.asarray([])
        azim_m = np.asarray([])
        elev_m = np.asarray([])
        quat_m = np.asarray([[]]).reshape((0,4))

        i=0
        for line in data:
            if i==0:
                i += 1
                continue
            else:

                words = line.split(';')

                words[4] = words[4].replace(']','')
                words[4] = words[4].replace('[','')

                time_m = np.append(time_m, np.asarray(float(words[0])))
                range_m = np.append(range_m, np.asarray(float(words[1])))
                azim_m = np.append(azim_m, np.asarray(float(words[2])))
                elev_m = np.append(elev_m, np.asarray(float(words[3])))

                quat_m = np.append(quat_m, np.expand_dims(np.fromstring(words[4], sep=' '), axis=0), axis=0)

                i += 1
                if i==100:
                    break

        if args['real_data_path'] is not None:
            real_data_path = os.path.join(args['real_data_path'], 'real_data.txt')

            with open(real_data_path, 'r') as save_file:
                data = save_file.readlines()

            time_r = np.asarray([])
            range_r = np.asarray([])
            azim_r = np.asarray([])
            elev_r = np.asarray([])
            quat_r = np.asarray([[]]).reshape((0, 4))

            i = 0
            for line in data:
                if i == 0:
                    i += 1
                    continue
                else:

                    words = line.split(';')

                    words[4] = words[4].replace(']', '')
                    words[4] = words[4].replace('[', '')

                    time_r = np.append(time_r, np.asarray(float(words[0])))
                    range_r = np.append(range_r, np.asarray(float(words[1])))
                    azim_r = np.append(azim_r, np.asarray(float(words[2])))
                    elev_r = np.append(elev_r, np.asarray(float(words[3])))

                    quat_r = np.append(quat_r, np.expand_dims(np.fromstring(words[4], sep=' '), axis=0), axis=0)

                    i += 1
                    if i == 100:
                        break

        time_m = time_m - time_m[0]
        time_r = time_r - time_r[0]

        range_coeffs = poly.polyfit(time_m, range_m, deg=3)
        azim_coeffs = poly.polyfit(time_m, azim_m, deg=3)
        elev_coeffs = poly.polyfit(time_m, elev_m, deg=3)

        #time_span = np.linspace(time_m[0], time_m[-1], 1000)

        range_fit = poly.polyval(time_m, range_coeffs)
        azim_fit = poly.polyval(time_m, azim_coeffs)
        elev_fit = poly.polyval(time_m, elev_coeffs)

        quat_diff = np.asarray([[]]).reshape((0,4))

        i=0
        for quat in quat_m:
            temp = np.asarray((quat[0], -quat[1], -quat[2], -quat[3]))
            temp = temp/(np.linalg.norm(temp)**2)
            temp_r = quat_r[i]
            temp_diff = np.expand_dims(np.asarray((temp[0]*temp_r[0] - temp[1]*temp_r[1] - temp[2]*temp_r[2] - temp[3]*temp_r[3],
                                    temp_r[0]*temp[1] + temp_r[1]*temp[0] - temp_r[2]*temp[3] + temp_r[3]*temp[2],
                                    temp_r[0] * temp[2] + temp_r[1] * temp[3] + temp_r[2] * temp[0] - temp_r[3] * temp[1],
                                    temp_r[0] * temp[3] - temp_r[1] * temp[2] + temp_r[2] * temp[1] + temp_r[3] * temp[0],
                                    )),axis=0)
            quat_diff = np.append(quat_diff, temp_diff,axis=0)
            i+=1



        plt.figure(figsize=(16,5))
        plt.subplot(1,3,1)
        plt.plot(time_m,range_m, 'b', label='measured')
        plt.plot(time_r, range_r, 'r', label='real')
        plt.plot(time_m, range_fit, '--', label='fit')
        plt.title('Range vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Range [m]')
        plt.ylim((0,2))
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(time_m, azim_m, 'b', label='measured')
        plt.plot(time_r, azim_r, 'r', label='real')
        plt.plot(time_m, azim_fit, '--', label='fit')
        plt.title('Azimuth vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Azimuth [deg]')
        plt.ylim((-20,20))
        plt.legend()


        plt.subplot(1,3,3)
        plt.plot(time_m, elev_m, 'b', label='measured')
        plt.plot(time_r, elev_r, 'r', label='real')
        plt.plot(time_m, elev_fit, '--', label='fit')
        plt.title('Elevation vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Elevation [deg]')
        plt.ylim((-20,20))
        plt.legend()

        plt.suptitle('Measured parameters')

        plt.figure(figsize=(16,5))
        plt.subplot(1,3,1)
        plt.plot(time_m, range_m - range_r, 'r', label='measured')
        plt.plot(time_m, range_fit -range_r, '--', label='fit')
        plt.title(r'Absolute Error : Range, $\mu = {:.3f}$, $\sigma^2={:.3f}$'.format(np.mean(np.abs(range_fit-range_r)),np.var(range_fit-range_r)))
        plt.xlabel('Time [s]')
        plt.ylabel('Error [m]')
        plt.ylim((-0.5,0.5))
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(time_m, azim_m - azim_r, 'r', label='measured')
        plt.plot(time_m, azim_fit - azim_r, '--', label='fit')
        plt.title(r'Absolute Error : Azimuth, $\mu = {:.3f}$, $\sigma^2={:.3f}$'.format(np.mean(np.abs(azim_fit-azim_r)),np.var(azim_fit-azim_r)))
        plt.xlabel('Time [s]')
        plt.ylabel('Error [deg]')
        plt.ylim((-10,10))
        plt.legend()


        plt.subplot(1,3,3)
        plt.plot(time_m, elev_m - elev_r, 'r', label='measured')
        plt.plot(time_m, elev_fit - elev_r, '--', label='fit')
        plt.title(r'Absolute Error: Elevation, $\mu = {:.3f}$, $\sigma^2={:.3f}$'.format(np.mean(np.abs(elev_fit-elev_r)),np.var(elev_fit-elev_r)))
        plt.xlabel('Time [s]')
        plt.ylabel('Error [deg]')
        plt.ylim((-10,10))
        plt.legend()

        plt.suptitle('Absolute errors')


        #plt.figure(figsize=(16,5))
        #
        #plt.subplot(1,3,1)
        #plt.plot(time_r, np.abs((range_m - range_r))/range_r*100, 'r', label='measured')
        #plt.plot(time_m, np.abs((range_fit -range_r))/range_r*100, '-.', label='fit')
        #plt.title('Relative Error : Range')
        #plt.xlabel('Time [s]')
        #plt.ylabel('Error [% of real value]')
        #plt.ylim((0,100))
        #plt.legend()
        #
        #plt.subplot(1,3,2)
        #plt.plot(time_r, np.abs((azim_m - azim_r)/azim_r)*100, 'r', label='measured')
        #plt.plot(time_m, np.abs((azim_fit -azim_r)/azim_r)*100, '-.', label='fit')
        #plt.title('Reative Error : Azimuth')
        #plt.xlabel('Time [s]')
        #plt.ylabel('Error [% of real value]')
        #plt.ylim(0,100)
        #plt.legend()
        #
        #plt.subplot(1,3,3)
        #plt.plot(time_r, np.abs((elev_m - elev_r)/elev_r)*100, 'r', label='measured')
        #plt.plot(time_m, np.abs((elev_fit -elev_r)/elev_r)*100, '-.', label='fit')
        #plt.title('Relative Error: Elevation')
        #plt.xlabel('Time [s]')
        #plt.ylabel('Error [% of real alue]')
        #plt.ylim(0,100)
        #plt.legend()
        #
        #plt.suptitle('Relative Errors')


        plt.figure(figsize=(16,5))
        plt.plot(time_r, quat_diff[:,0],'b', label='qw')
        plt.plot(time_r, quat_diff[:,1],'r', label='qx')
        plt.plot(time_r, quat_diff[:,2],'m', label='qy')
        plt.plot(time_r, quat_diff[:,3],'--', label='qz')
        plt.axhline(y=1, color='0', linestyle='-')
        plt.axhline(y=np.sqrt(2)/2, color='0', linestyle='-')
        plt.axhline(y=0.5, color='0', linestyle='-')
        plt.axhline(y=0, color='0', linestyle='-')
        plt.axhline(y=-np.sqrt(2)/2, color='0', linestyle='-')
        plt.axhline(y=-0.5, color='0', linestyle='-')
        plt.axhline(y=-1, color='0', linestyle='-')

        plt.xlabel('Time [s]')
        plt.ylabel('Quaternion difference')
        plt.ylim((-1.2,1.2))
        plt.legend()
        plt.title('Quaternion difference')

        plt.show()

        #print(quat)


if __name__ == '__main__':
    main()

