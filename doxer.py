#! /usr/bin/python3

"""
     █████                                        
    ░░███                                         
  ███████   ██████  █████ █████  ██████  ████████ 
 ███░░███  ███░░███░░███ ░░███  ███░░███░░███░░███
░███ ░███ ░███ ░███ ░░░█████░  ░███████  ░███ ░░░ 
░███ ░███ ░███ ░███  ███░░░███ ░███░░░   ░███     
░░████████░░██████  █████ █████░░██████  █████    
 ░░░░░░░░  ░░░░░░  ░░░░░ ░░░░░  ░░░░░░  ░░░░░     
                                                  
Doxer Stylometric Data Mining Library v1.3
By Troy J. Watson

Example:

Basic 1-gram word search for Satoshi 
doxer.py -t 3satoshi

Find the author for Satoshi with 4 ngrams using characters
doxer.py -n 4 -c -t 3satoshi

Reduce massive dataset with forest then classify: 
doxer.py -t 3satoshi -f 

Do the same again only with character n-grams with forest:
doxer.py -t 3satoshi -f -c

Reset pickle of table and run a fresh forest:
doxer-py -t 3satoshi -f --reset

"""
import argparse
import os
import sys
import glob
import math
import random
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class Doxer(object):
	def __init__(self):
		pass

	def split(self, ls):
		'''
		useage: 
		split([0 for x in range(12)])
		creates constant skip grams
		'''
		if len(ls) < 4: return [1 for x in range(len(ls))]
		if len(ls) == 5: return [1,1,0,1,1]
		m = math.ceil(len(ls)/2)
		m1 = math.ceil(m/2)
		m2 = math.ceil(m/2) + m - 1
		ls[m1] = 1; ls[m2] = 1; ls[0] = 1; ls[-1] = 1
		return ls

	def testSkipGram(self):
		skip = self.split([0 for x in range(10)])
		tst = [1,2,3,4,5,6,7,8,9,10]
		out = [y[1] for y in zip(skip,tst) if y[0] == 1]
		return out == [1,4,8,10]

	def testWordGram(self):
		tst = ['here','is','a','test']
		out1 = self.ngrams(tst,2)
		out2 = self.ngrams(tst,3)
		print(self.ngrams('here are some words to test a here now'.split(' '), 6))
		res1 = [['here','is'],['is','a'],['a','test']]
		res2 = [['here','is','a'],['is','a','test']] 
		return out1 == res1 and out2 == res2

	def ngrams(self,l, n = 4):
		c = 0
		out = []
		skip = self.split([0 for x in range(n)])
		while True:
			if n >= 4:
				tmp = [y[1] for y in zip(skip,l[c:c+n]) if y[0] == 1]
				out += [tmp]
			else:
				out += [l[c:c+n]]
			c += 1
			if c >= len(l) - n + 1:
				break
		return out

	def splitText(self, s, CHAR=False):
		if CHAR:
			return list(s)
		else:
			return s.split(' ')

	def ngramDict(self, l, CHAR=False):
		d = {}
		for x in l:
			# I added this to keep spaces for word grams
			if CHAR:
				xx = ''.join(x)
			else:
				# These spaces allow for post mortem analysis of where the author uses certain words... great for creating a graphical display. 
				xx = ' '.join(x)
			if xx in d:
				d[xx] += 1
			else:
				d[xx] = 1
		return d

	def processText(self,txt,CHAR,NGRAM):
		t = open(txt).read()
		t = self.splitText(t,CHAR)

		return self.ngramDict(self.ngrams(t,NGRAM),CHAR)

	def analyse(self, fs, CHAR=False, NGRAM=1,AUTHOR='Satoshi',OUTPUT=False):
		# Masterkey counts n documents have ngram pattern
		masterkey = {}
		# Finalkey stores ngram overlap between candidate,author,and other candidates
		finalkey = {}
		# Added set() to get rid of duplicates... if there are duplicate filenames then it will crash 
		for f in set(fs):
			# Init finalkey format for each candidate
			# [0,{}] -> 0 will be count of overlap between this candidate and test text, {} will store {'candidate_i': n} for overlap with all other candidates
			CURRENT_AUTHOR = ''.join(f.split('_')[0:-1]) 
			if CURRENT_AUTHOR != AUTHOR:
				finalkey[''.join(f.split('_')[0:-1])] = [0,{}]
			for ff in fs:
				# Skip the test author
				if ''.join(ff.split('_')[0:-1]) == AUTHOR:continue
				# ('_')[0:-1] is just extracting author name from filename
				# Init {'candidate_i':n} structure mentioned earlier 
				if CURRENT_AUTHOR != AUTHOR and ''.join(ff.split('_')[0:-1]) != CURRENT_AUTHOR:
					finalkey[''.join(f.split('_')[0:-1])][1][''.join(ff.split('_')[0:-1])] = 0
			# Load candidate document and convert it into a dict of ngrams
			doc_dict = self.processText(f,CHAR,NGRAM)

			# Add dict entries to master key
			for x,y in doc_dict.items():
				if x in masterkey:
					# If masterkey is already above 2 different candidates containing ngram then just +=1 because there's no need to track which candidates overlap ( format = {ngram : [n} ) 	
					if len(masterkey[x]) < 2:
						masterkey[x][0] += 1
					else:
						# If masterky item has not yet been pruned, then carry on as normal, iterating first element and appending authors name to list
						masterkey[x][0] += 1
						# ('_')[0:-1] just extracts authors name
						masterkey[x][1] += [''.join(f.split('_')[0:-1])]
						# Yet if number of candidates overlaping is above 2 then we don't need to list candidates names because this entry is not going to be used later
						# We keep the placeholder of this entry however because we need to know that it exists so as to not count it as an overlap between two documents
						if masterkey[x][0] > 2:
							# This will be new format for ngrams that have more than 2 candidates using the word/characters { ngram : [n] } 
							masterkey[x]=[masterkey[x][0]]
				# This will be the first place you go to count ngrams because at the start all ngrams have 0 documents in masterkey
				else:
					'''
					masterkey format:
					{ ngram : [docs, [name1,name2]] } 
					'''
					# The format is { ngram : [ docs, [name1,name2]] }
					# e.g. { 'the' : [2, ['author1','author2']] } means that only author1 and author2 (so far) have used the word 'the'
					# if more than 2 authors use the word 'the' then it will be pruned to { 'the' : [3] } ... but that pruning happens at the top of this IF statement
					masterkey[x] = [1, [''.join(f.split('_')[0:-1])]] 
		# Now that we have a complete masterkey we can count the overlap between the candidates and the test document
		for x,y in masterkey.items():
			# len(y) < 2 means the ngram has been pruned such as in the item { 'the', [25] } meaning all 25 items in the corpus use the word 'the' and thus it will be ignored
			if len(y) < 2: continue
			# Ignore items with more than 2 overlap
			if y[0] > 2: continue
			# Ignore items where only 1 document used the ngram
			if y[0] == 1: continue
			# Keep a counter simply to rotate the candidates used (there should only be 2 in item that we're viewing)
			c = 0
			for l in y[1]:
				# if c is up to 0 then the opposite of that is 1, this is just a latch function to rotate between the two items being compared
				if c == 0:
					ll = 1
				else:
					ll = 0

				if AUTHOR in y[1]:
					if l == AUTHOR: 
						c+=1
						continue
					# If you're looking at a candidate that coincides with the test doc, update the overlap in the finalkey. 
					'''
					finalkey format:
					{'candidate': [n, {'author1':n1, 'author2':n2}]}
					'''
					# Reminder, the finalkey format is {'candidate': [ n, {'author1': n1, 'author2': n2}] } where n = overlap with test doc (e.g. satoshi) and n1 is the overlap with author1. We need the overlap with author1 and author2 because later we will compare the overlap between these other texts as against the test text. The idea is that we would expect the test text to overlap more than the other texts in the corpus if they are the same author. 
					finalkey[l][0] += 1 
				else:
					# This is where we use the flip function on overlaps that don't involve the test doc (e.g. satoshi)... we add these overlaps to the author1 or author2 entries in {'candidate': [n, {'author1':n1, 'author2':n2}]} 
					kk = y[1][ll]
					finalkey[l][1][kk] += 1
				c+=1
		# Create a final list to sort the best candidates to the top
		finallist = []
		for x,y in finalkey.items():
			if x == AUTHOR: continue
			tmp = []
			for xx,yy in y[1].items():
				tmp += [yy]
			finallist += [[y[0] / (sum(tmp)/len(tmp)+0.00001),x]]
			#finallist += [[y[0] / max(tmp),x]]
		finallist = sorted(finallist,reverse=True)
		#TODO prints final list for winner
		#print(finalkey[finallist[0][1]])
		#TODO lists final top 20 score
		#for l in finallist[:20]:
		#	print('{} , {}'.format(l[1],l[0]))
		#{ ngram : [docs, [name1,name2]] } 
		for x,y in masterkey.items():
			if len(y) < 2 : continue
			if sorted(y[1]) == sorted([finallist[0][1], AUTHOR]):
				#TODO this is printing the words that overlap
				if OUTPUT: print(x)
		#TODO return basic result
		return finallist[0][1]
			
	
"""
Main section

grep -l back-of-the-envelope * 
"""
def main(args):
	doxer = Doxer()

	def preprocess(data,q,save=False):
		'''
		Note: the whole point of this preprocessing is to reduce a large folder size down to the most likely candidates using Burrows Delta (without the z-scores). This approach to reduction can be replaced with a more powerful method later on and will be essential to uncovering the identity of obscure texts. 
		'''
		if save: masterkey = {}
		# Get ngram dict for Q document
		qd = doxer.processText(q[0],args.charGram,args.ngram)
		# Store top ngrams in a list 
		ql = []
		# Create a list of all Q items with tally at front for sorting
		for x,y in qd.items():
			ql += [[y,x]]
		# Sort the ngrams by largest first and cut off top 100 for quick reduction of dataset
		ql = sorted(ql,reverse=True)[:1000]

		# Get the total N of grams in top 100
		qn = sum([x[0] for x in ql])
		# Get term frequency for each gram
		qf = [[x[0]/(qn+0.00000001),x[1]] for x in ql]
		# Create Q dict for mapping { ngram : frequency } 
		qd = {}
		for x in qf:
			qd[x[1]] = x[0]	
		# Corpus list will store top documents to later analyse with Doxer
		if save: masterkey[q[0]] = qd
		corpus = []
		for f in data:
			# ignore Q text in loop over folder
			if ''.join(f.split('_')[:-1]) == args.testText:
				continue
			# get word grams for each candidate in folder
			d = doxer.processText(f,args.charGram,args.ngram)
			# Just copy Q dict and zero out all entries to reuse it
			dd = qd.copy()
			for x,y in dd.items():
				dd[x] = 0
			for x,y in d.items():
				if x in dd:
					dd[x] = y
			# Get total N of ngrams in candidate dict
			n = sum([x[1] for x in dd.items()])
			# Calculate Burrows' Delta distance between Q doc and candidate doc
			if save: masterkey[f] = dd
			#FIXME qd.items() is probably bigger than dd{}
			#TODO normalize the data otherwise the first items will disproportionately affect all of the weights
			delta = sum([abs(x[1] - dd[x[0]]/(n+0.00000001)) for x in qd.items()]) / len(qd)
			#TODO corpus += [ [x[1] for x in qd.items()] ] 

		
			
			# Add the delta value and filename to corpus list to later pick the best results out closer to Q
			corpus += [[delta,f]]
		# Here is where we sort the delta list and cut off anything over our -r number requested
		corpus = sorted(corpus,reverse=False)[:args.reduce]
		# Print out the results 
		#TODO print out reduced list here
		#for x in corpus:
		#	print('{} -> {}'.format(x[0],x[1]))
		# Add the Q filename to corpus list so that it can be analysed by Doxer
		corpus = [x[1] for x in corpus] + q
		if save: return corpus, masterkey
		return corpus


	if args.unitTest:
		print('Skip Grams... {}'.format(doxer.testSkipGram()))
		print('Word Grams... {}'.format(doxer.testWordGram()))
	
	elif args.forest:
		##########################################
		# STEP 1 : COLLECT FREQUENCIES 
		##########################################
		
		def ngrams(l, n = 4):
			c = 0
			out = []
			while True:
				out += [l[c:c+n]]
				c += 1
				if c >= len(l) - n + 1:
					break
			return out

		def ngramDict(l):
			d = {}
			for x in l:
				xx = ''.join(x)
				if xx in d:
					d[xx] += 1
				else:
					d[xx] = 1
			return d

		def processText(txt,NGRAM=4,word=False):
			t = open(txt).read().lower()
			#NOTE use these if using words
			if word: 
				r = t.split(' ')
				t = []
				for rr in r:
					t += [''.join(list([val for val in rr if val.isalpha()]))]
			else:
				#NOTE use this if using ngrams
				t = ''.join(list([val for val in t if val.isalpha()]))
			return ngramDict(ngrams(t,NGRAM))

		if args.charGram:
			WORD = False
		else:
			WORD = True


		if args.gut:
			KIN = False
		else:
			KIN = True

		if WORD:
			NGRAM = 1
		else:
			NGRAM = 4

		if WORD: w = 'w'
		else: w = 'c'

		buff_title = 'table-' + str(w) + '.pickle'

		buff = True	
		try: 
			zscore = pickle.load(open(buff_title,'rb'))
		except:
			buff = False	

		if args.reset:
			buff = False
			
		if WORD:
			ff = open('/home/flak/Documents/prog/.git/doxer/1w-300a/masterkey.pickle','rb')
		else:
			ff = open('/home/flak/Documents/prog/.git/doxer/4g-1000a/masterkey.pickle','rb')
		ll = pickle.load(ff)


		if not(buff):
			tags = [l[1] for l in ll]

			d = {}
			for l in ll:
				d[l[1]] = 0

			fs = glob.glob('*.txt')

			d_sum = d.copy()
			n_sum = 0

			zscore = {}

			for f in fs:
				print(f.split('/')[-1])
				dd = d.copy()

				ngs = processText(f,NGRAM,WORD)
				n = 0
				for t in tags:
					if t in ngs:
						n += ngs[t]
					else:
						ngs[t] = 0

				for t in tags:
					try:
						dd[t] = ngs[t] / n 
						d_sum[t] += ngs[t] / n
					except:
						print('error divide by zero')
						
				out = [dd[t] for t in tags]
				
				title = f.split('/')[-1].split('.')[0]
				zscore[title] = out	

				n_sum += 1
			with open(buff_title,'wb') as f:
				pickle.dump(zscore, f)	


		##########################################
		# STEP 2 : CREATE PAIRWISE COMPARISONS 
		##########################################
		KEYWORD = args.testText
		#NOTE create X for forest
		X = []
		yy = []

		ll = []
		for x,y in zscore.items():
			ll += [x]
		l = []
		d = {}
		cx = 0
		cy = 0
		true_c = 0
		for x in ll:
			cy = 0
			for y in ll:
				if x==y:
					pass
				elif x.split('_')[0] != KEYWORD and y.split('_')[0] != KEYWORD:
					pass
				else:
					l = sorted([cx,cy])
					if l[0] in d:
						if l[1] in d[l[0]]:
							cy += 1
							continue
						else:
							d[l[0]] += [l[1]]
					else:
						d[l[0]] = [l[1]]
					txt1 = zscore[ll[l[0]]]
					txt2 = zscore[ll[l[1]]]
					txts = list(zip(txt1,txt2))
					delta = [abs(x[0] - x[1]) for x in txts]
					yy += [[x,y]]
					X += [delta]
					true_c += 1
				cy += 1
			cx += 1
		print(X[0])
		print(yy[0])

		##########################################
		# STEP 3 : RUN DATA ON FOREST MODEL 
		##########################################

		if KIN:
			knn = 'kin'
		else:
			knn = 'gut'

		if WORD: 
			nns = [100,200,300]
			gms = '1w'
		else:
			nns = [100,200,300,400,500,600,700,800,900,1000]
			gms = '4g'

		def trans(l): return list(map(list, zip(*l)))
		# Collect all reduced candidates here
		r = []
		print('Running over all Forest models')
		print('--------------------------------------')
		for code in nns:	
			ss = gms + '-rs-' + knn + '-' + str(code) + 'a.pickle'
			fn = '/home/flak/Documents/prog/.git/forest/'+ss	
			if WORD and code == 300:
				codes = 298
			else:
				codes = code
			tmpX = trans(trans(X)[:codes])

			loaded_model = pickle.load(open(fn,'rb'))
			predictions = loaded_model.predict(tmpX)
			keys = yy
			c = 0
			for p in predictions:
				if int(p) > 0:
					print(c,p,keys[c])
					for k in keys[c]:
						r += [k]
				c+=1

		##########################################
		# STEP 4 : RUN DOXER ON REDUCED LIST 
		##########################################

		# Get list of files in folder to test against Q

		ddd = {}
		for rr in r:
			if rr in ddd:
				ddd[rr] += 1
			else:
				ddd[rr] = 1
		r = sorted([[ddd[x],x]for x in set(r)], reverse=True)[:10]
		print(r)
		data = [x[1] + '.txt' for x in r]
		print()
		print('Reduced list of forest candidates:')	
		print('--------------------------------------')
		# Q document is the document in question
		q = glob.glob(args.testText + '*.txt')

		if args.forestOnly:
			sys.exit()

		print()
		print('Analyzing Word Grams 1 - 10')
		print('--------------------------------------')
		for y in range(1,10):
			fk = doxer.analyse(fs=data,CHAR=False,NGRAM=y,AUTHOR=args.testText, OUTPUT=args.output)
			print(y,fk)

		print()
		print('Analyzing Char Grams 4 - 10')
		print('--------------------------------------')
		for y in range(4,10):
			fk = doxer.analyse(fs=data,CHAR=True,NGRAM=y,AUTHOR=args.testText, OUTPUT=args.output)
			print(y,fk)


	elif args.pan:

		#q = ['GR009unknown_greek.txt']
		#data = ['GR009known_greek.txt','GR012unknown_greek.txt']
		#TODO loop over this with multiple conditions
		fs = glob.glob('*.txt')
		score = 0
		tt = 0
		for f in fs:
			if f.split('-')[-1].split('_')[0] == 'known': continue
			k = f.split('-')[0]
			o = glob.glob(k + '-known_*')
			c = 1
			fo = [ff for ff in fs if ff != f and ff != o[0]]
			vote = []
			for x in range(10):
				random.seed(c)
				oo = o + []
				oo += [random.sample(fo, 10)[0]]
				#corpus = preprocess(o,[f])
				oo += [f]
				fk = doxer.analyse(fs=oo,CHAR=args.charGram,NGRAM=args.ngram,AUTHOR=f.split('_')[0])
				vote += [fk.split('-')[0] == f.split('-')[0]]
				c += 1
			if sum(vote) > 5:
				cl = True
			else:
				cl = False
			act = f.split('_')[-1].split('.')[0] == 'True'  
			print(cl,act)
			if cl == act:
				score += 1
			tt += 1
		print(score,tt,score/tt)

	elif args.benchmark:
		# English : 0.83
		# Russian : 0.8518518518518519 
		# Polish : 0.85

		#TODO 1) get top 1000 word grams over folder
		#TODO 2) loop over texts
		#TODO 3) get 10 knn for doxer
		fs = glob.glob('*.txt')
		#_,masterkey = preprocess(fs[1:],[fs[0]],save=True)
		count = 0; total = 0
		for f in fs:	
			#q = masterkey[f]
			#dd = {}
			#for a,b in masterkey.items():
			#	if a == f: continue
			#	n = sum([x[1] for x in b.items()])
			#	delta = sum([abs(x[1] - b[x[0]]/(n+0.00000001)) for x in q.items()]) / len(q)
			#	dd[a] = delta
			#ls = sorted([[x[1],x[0]] for x in dd.items()], reverse=False)[:args.reduce]
			#corpus = [x[1] for x in ls]
			fs2 = [ff for ff in fs if ff != f]
			corpus = preprocess(fs2,[f])
			fk = doxer.analyse(fs=corpus,CHAR=args.charGram,NGRAM=args.ngram,AUTHOR=f.split('_')[0])
			A = f.split('-')[0]
			B = fk.split('-')[0]
			print( A == B)
			if A==B: 
				count += 1
			total += 1
		print(count,total,count/total)
	else:
		# Get list of files in folder to test against Q
		data = glob.glob('*.txt')
		# Q document is the document in question
		q = glob.glob(args.testText + '*.txt')
		# Reduce the large folder down to a manageable list
		corpus = preprocess(data,q)
		# Run Doxer!
		fk = doxer.analyse(fs=corpus,CHAR=args.charGram,NGRAM=args.ngram,AUTHOR=args.testText, OUTPUT=args.output)
		print(fk)
		
if __name__ == '__main__':	   
	"""
	Argparse section
	"""
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description=__doc__)

	parser.add_argument('-n',help='Number of ngrams (default: 1)', type=int,default=1,dest='ngram')
	parser.add_argument('-r',help='Reduce dataset down to a reasonable number (default: 50)', type=int,default=50,dest='reduce')
	parser.add_argument('-c',help='Use Character grams instead of Word grams (default: False)',dest='charGram',action='store_true')
	parser.add_argument('-t',help='Test author (default: Satoshi)',type=str,default='Satoshi',dest='testText')
	parser.add_argument('-u',help='Run UnitTest on functions',dest='unitTest',action='store_true')
	parser.add_argument('-b',help='Benchmark folder of texts',dest='benchmark',action='store_true')
	#TODO add v for verbose
	parser.add_argument('-p',help='Pan 2014 Benchmark over folder',dest='pan',action='store_true')
	parser.add_argument('-o',help='Output exclusive ngrams used by author and classified candidate',dest='output',action='store_true')
	parser.add_argument('-g',help='Use Gutenberg models for forest',dest='gut',action='store_true')
	parser.add_argument('-f',help='Run forest model',dest='forest',action='store_true')
	parser.add_argument('-ff',help='Only run forest model and quit before running doxer',dest='forestOnly',action='store_true')
	parser.add_argument('--reset',help='Reset pickle of tables',dest='reset',action='store_true')

	args = parser.parse_args()

	main(args)
