import os
import time

cp_freq = os.popen('adb shell cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies').read().strip().split(' ')
if not os.path.exists('logs'):
    os.makedirs('logs')

for cp_f in cp_freq:
    print "CP Frequency: " + str(cp_f)
    time.sleep(1)
    os.system('adb shell \"echo ' + str(cp_f) + ' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq\"')
    os.system('adb shell \"echo ' + str(cp_f) + ' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq\"')
    os.system('/home/mallesh/Desktop/Mallesh/PLTSpeed/live_test_mallesh.py')
    os.system('cp -r /home/mallesh/Desktop/Mallesh/PLTSpeed/desktop-compression_uist-img logs/'+str(cp_f))
    time.sleep(1)
