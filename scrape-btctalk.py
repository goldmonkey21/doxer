# Load Random User Agent
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
import requests
import sys
import csv 
from bs4 import BeautifulSoup
import os

software_names = [SoftwareName.CHROME.value]
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]   
user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
user_agents = user_agent_rotator.get_user_agents()
user_agent = user_agent_rotator.get_random_user_agent()

url = 'https://bitcointalk.org/index.php?action=profile;u='
gavin = url + '224'
satoshi = url + '3'
posts = ';sa=showPosts'
postPage = ';start='
gavin_posts = url + gavin + posts

#################################################
# STEP 1 : GET LIST OF PROFILE IDS
#################################################
'''
profiles = []
c = 0
cc = 0
#TODO change range to where you are up to
for x in range(1001,2001):
	user_agents = user_agent_rotator.get_user_agents()
	r = requests.get(url + str(x))
	b = BeautifulSoup(r.content,'html.parser').title.text
	if b.split(' ')[-1] == 'Occurred!': 
		c += 1
		cc += 1
		continue
	print('{},{}'.format(x,' '.join(b.split(' ')[4:])))
	profiles += [[x,' '.join(b.split(' ')[4:])]]
	if c > 50:
		with open('output_'+str(cc)+'.csv','w') as f:
			write = csv.writer(f)
			write.writerows(profiles)
		os.system('nordvpn d')
		os.system('nordvpn c')
		c = 0
	c += 1
	cc += 1
print(profiles)
with open('output_'+str(cc)+'.csv','w') as f:
	write = csv.writer(f)
	write.writerows(profiles)
sys.exit()
'''

#################################################
# STEP 2 : SAVE TEXT OF ALL PROFILES IN LIST 
#################################################

urls = []
txt = []
def thread2text(u):
	r = requests.get(u)
	b = BeautifulSoup(r.text, 'html.parser')
	a = b.find_all('a',class_='navPages')
	for e in b.find_all('div',class_='quote'):
		e.extract()
	for e in b.find_all('div',class_='quoteheader'):
		e.extract()
	return ' '.join([x.text for x in b.find_all('div',class_='post')]),a

def changeIP():
	os.system('nordvpn d')
	os.system('nordvpn c')

cc = 0
#TODO change csv file for latest one
f = 'top1001-2000.csv'
with open(f,'r') as data:
	for line in csv.reader(data):
		#NOTE use this to download one user:
		#line = ['101601', 'adam3us']
		txt = []
		user_agents = user_agent_rotator.get_user_agents()
		if cc > 50:
			changeIP()
			cc = 0
		t,a = thread2text(url + line[0] + posts)
		if len(t) == 0 and len(a) == 0:
			cc += 1
			continue
		txt += [t]
		if len(a) > 0:
			last_page = int(a[-2].text) * 20 - 20 
			urls = [url + line[0] + posts + postPage + str(x) for x in range(20,last_page,20)]
			for u in urls:
				print(u)
				if cc > 50:
					changeIP()
					cc = 0 
				t,_ = thread2text(u)
				txt += [t]
				cc += 1
		fn = str(line[0]) + str(line[1]) + '_forum.txt'
		print('Writing to file: {}'.format(fn))
		with open(fn,'w') as f:
			for l in txt:
				f.write(l)
				f.write('\n')
		cc += 1
		#NOTE use this to download one user:
		#sys.exit()
