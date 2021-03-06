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
                                                  
Doxer Stylometric Data Mining Library v1.5
By Troy J. Watson

Example:

Basic 1-gram word search for Satoshi 
doxer.py -t 3satoshi -r 10 

Find the author for Satoshi with 4 ngrams using characters
doxer.py -n 4 -c -t 3satoshi -r 10 

"""
import argparse
import os
import sys
import glob
import math
import random

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

	def analyse(self, fs, CHAR=False, NGRAM=1,AUTHOR='Satoshi',OUTPUT=False,VERBOSE=False):
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
			#finallist += [[y[0] / (sum(tmp)/len(tmp)+0.00001),x]]
			if y[0] == 0:
				continue
			finallist += [[y[0] / (max(tmp)+0.00001),x]]
		finallist = sorted(finallist,reverse=True)
		if VERBOSE:
			for l in finallist[:20]:
				print('{} , {}'.format(l[1],l[0]))
		#{ ngram : [docs, [name1,name2]] } 
		for x,y in masterkey.items():
			if len(y) < 2 : continue
			if sorted(y[1]) == sorted([finallist[0][1], AUTHOR]):
				if OUTPUT: print(x)
		return finallist[0][1]
			
	
"""
Main section
"""
def main(args):
	doxer = Doxer()

	def preprocess(data,q,save=False):
		if save: masterkey = {}
		# Get ngram dict for Q document
		qd = doxer.processText(q[0],False,1)
		# Store top ngrams in a list 
		ql = []
		# Create a list of all Q items with tally at front for sorting
		for x,y in qd.items():
			ql += [[y,x]]
		# Sort the ngrams by largest first and cut off top 300 for quick reduction of dataset
		ql = sorted(ql,reverse=True)[:300]

		# Get the total N of grams in top 300
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
			d = doxer.processText(f,False,1)
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
			#TODO normalize the data otherwise the first items will disproportionately affect all of the weights
			delta = sum([abs(x[1] - dd[x[0]]/(n+0.00000001)) for x in qd.items()]) / len(qd)
			
			# Add the delta value and filename to corpus list to later pick the best results out closer to Q
			corpus += [[delta,f]]
		# Here is where we sort the delta list and cut off anything over our -r number requested
		corpus = sorted(corpus,reverse=False)[:args.reduce]
		def _reduce(ls):
			c = 0 ; t = []
			ls = [l[0] for l in ls]
			for l in ls:
				if c == 0: pass
				else:
					t += [ls[c] - ls[c-1]]
				c += 1
			return t

		def _split(ls):
			c = 0 ; m = []
			def _mean(x):
				return sum(x) / len(x)
			for l in ls:
				#if c == 0: pass
				#else:
				m += [[l, c+1]]
				c += 1
			#print(sorted(m))
			tmp = 99
			for ll in m:
				if ll[0] < tmp:
					tmp = ll[0]
				else:
					#NOTE its returning item with length too long
					#NOTE but [:ll[1]] cuts it out in corpus
					return ll[1]
			return sorted(m,reverse=False)[0][1]

		#TODO make split point with gini function
		#print(_reduce(corpus))

		corpus = corpus[:_split(_reduce(corpus))]

		# Print out the results 
		for x in corpus:
			print('{} -> {}'.format(x[0],x[1]))
		# Add the Q filename to corpus list so that it can be analysed by Doxer
		corpus = [x[1] for x in corpus] + q
		if save: return corpus, masterkey
		return corpus


	if args.unitTest:
		print('Skip Grams... {}'.format(doxer.testSkipGram()))
		print('Word Grams... {}'.format(doxer.testWordGram()))
	
	elif args.benchmark:
		# English novels : 80%
		# Russian novels: : 85.18%
		# Polish novels : 77%

		fs = glob.glob('*.txt')
		count = 0; total = 0 ; f_count = 0
		for f in fs:	
			fs2 = [ff for ff in fs if ff != f]
			corpus = preprocess(fs2,[f])
			if len(corpus) > 2:
				def _count(ls):
					d = {}
					for l in ls:
						if l in d:
							d[l] += 1
						else:
							d[l] = 1
					out = []
					for k,v in d.items():
						if v > 1:
							out += [k]
					return out
				txts = _count([x.split('-')[0] for x in corpus])
				if len(txts) == 1:
					fk = txts[0] + '-text'
				else:
					fk = doxer.analyse(fs=corpus,CHAR=args.charGram,NGRAM=args.ngram,AUTHOR=f.split('_')[0],VERBOSE=args.verbose)
			else:
				fk = [x for x in corpus if x != f][0]
			A = f.split('-')[0]
			B = fk.split('-')[0]
			print(A)
			if A == 'Anon':
				A = 'Blackmore'
			print( A == B)
			if A==B: 
				count += 1
			else:
				f_count += 1
			total += 1
			print(f_count / total) 
		print(count,total,count/total)
	else:
		# Get list of files in folder to test against Q
		data = glob.glob('*.txt')
		# Q document is the document in question
		q = glob.glob(args.testText + '*.txt')
		# Reduce the large folder down to a manageable list
		corpus = preprocess(data,q)
		# Run Doxer!
		if len(corpus) > 2:
			fk = doxer.analyse(fs=corpus,CHAR=args.charGram,NGRAM=args.ngram,AUTHOR=args.testText, OUTPUT=args.output,VERBOSE=args.verbose)
		else:
			fk = corpus
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
	parser.add_argument('-v',help='Verbose output to analyse all of the steps',dest='verbose',action='store_true')
	parser.add_argument('-o',help='Output exclusive ngrams used by author and classified candidate',dest='output',action='store_true')

	args = parser.parse_args()

	main(args)
