import os
import time

if not os.path.exists('logs'):
	os.makedirs('logs')

categories = ['kidsandteens.txt', 'news.txt', 'sports.txt', 'health.txt', 'shopping.txt']

for cat in categories:
	os.system('cp websites/'+cat+' res/mixed_live_100runs.txt')
	os.system('sudo /home/mallesh/Desktop/Mallesh/PLTSpeed/live_test_mallesh.py')
	os.system('cp -r /home/mallesh/Desktop/Mallesh/PLTSpeed/desktop-compression_uist-img logs/'+str(cat[:-4]))
