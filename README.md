# Doxer Stylometric Data Mining Library 🕵️


![Agatha Christie quote- very few of us are what we seem](images/agatha-christie-quote.png)

## Quick-start 

Clone this repository: 

```git clone https://github.com/goldmonkey21/doxer```

Add doxer to your ```.bashrc``` file: 

```
cd ~
sudo vim .bashrc
export PATH=$PATH:/home/flak/Documents/prog/.git/doxer
```

Find Satoshi identity in btc folder:

```
cd btcforum
doxer.py -t 3satoshi
```

Output for Satoshi:

```
0.0010055437039908227 -> 237lachesis_forum.txt                      
0.001015254105998512 -> 224GavinAndresen_forum.txt
224GavinAndresen
```

👍 Australian born programmer Gavin Andresen is Satoshi!


Find author of original Q-source gospel:

```
cd christiantexts
doxer.py -t qsource
```

Output for Q-source:

```
0.001427588237734938 -> matthew-web_christian.txt 
0.0015145750641581285 -> luke-web_christian.txt
0.0015699158741751618 -> thomas-layton_christian.txt
thomas-layton
```

👍 Gospel of Thomas and original Christian Gospel were written by a Gnostic!


Find author of Daniel (old testament text):

```
cd lxx
doxer.py -t Daniel
```

Output for Daniel:

```
0.0012709272493254052 -> KingsI_lxx.txt
0.001307254569569858 -> Genesis_lxx.txt
Genesis
```

👍 Both books of Daniel and Genesis written by same person, pushing date of entire old testament to Hellenistic era!


Find anonymous novel ```Clara``` in benchmark:

```
cd novels_english
doxer.py -t Anon-Clara1864
```

Output for Clara novel:

```
0.000670058804150311 -> Blackmore-Lorna1869_english.txt
0.0007114990852890783 -> Blackmore-Erema1877_english.txt
0.0007436772047638017 -> Cbronte-Jane1847_english.txt
0.0007649839175935351 -> Cbronte-Villette1853_english.txt
Blackmore-Lorna1869
```

👍 And that is the correct answer... Blackmore wrote the novels Clara and Lorna!

Done!

## Introduction

![Simple Stylometry in Terminal](images/general-doxer-useage.png)

Let us start this research project with a quick word about Agatha Christie (as pictured in the quote above). I have always likened my work as a data miner to that of Christie's most famous character Miss Marple. Far from being a lone spinster, Miss Marple is able to outwit some of the most clever of criminals simply because she has read enough crime books to gain a somewhat sixth sense into their goings on. With that I will encourage you to take my argument seriously and to even install my software on your own computer. The Python library after all is self contained and needs not many extra imports. With that said, let's uncover the identity of Satoshi and maybe learn a few more lessons about data mining along the way. 

What I wanted more than anything else is a stylometry program that could run easily from my Linux terminal and gather robust stylometric results without the need for a GPU. I lay feverishly in my bed trying to solve this problem while also surviving a cold. I came up with an algorithm that when I awoke immediately translated into code so as to solve most of this problem once and for all. 

If you pay attention to the above diagram, you will see that Doxer is able to find the identity of the famous Russian writer Gorky by simply typing `doxer.py -t Gorky-Mat` The result is an immediate match without much work being put into it. The algorithm thus is unsupervised and also can work anywhere on terminal if you add it to your `.bashrc` file. 

And by way of embarrassment I admit that I made a typo in the image above, writing 200 instead of 2,000. I analyzed all substantial texts between 1 and 2,000 on the bitcoin forum, adding an extra Adam Back just for the fans of his authorship. 

Moving on to another line in the diagram above, you can see Doxer identifying a text labeled as Anon-Clara1864. This file-name means that it is titled Clara and was published in 1864. Doxer immediately uncovers the author's identity and correctly attributes Blackmore as the culprit. Here is the wiki page about the book which you can check for yourself: 

https://en.wikipedia.org/wiki/Clara_Vaughan


This program can be used for a multitude of stylometry tasks, but I will contain the scope of this case study to the mystery of Satoshi Nakamoto, the inventor of Bitcoin. I will use Doxer to finalize my analysis but I will additionally employ a Random Forest on an Amazon EC2 instance to reduce the list of candidates down to a manageable (yet reasonable) amount. 

In a short word, Doxer is a unique word analyzer which takes upon itself the task of finding all of the unique words that two texts share. You first take an unknown text and compare it one by one with all of the other possible candidate texts. Slowly but surely you count how many times each candidate text shares a word with only the unknown text. For example, the texts by Satoshi and Gavin may use particular words (or ngrams) that no other text in the corpus uses. If this number divided by the average-overlap-between Gavin-and-all-other-texts is highest among all candidates then it would be reasonable to suggest that out of those candidates the most likely author is Gavin himself. Of course, the algorithm will not work as well if you feed it a million texts because the overlap of words will be dispersed over the entire corpus. It is therefore necessary to reduce the corpus first so as to only analyze those texts that are already similar in style to the unknown text. 

Let's give you a quick toy example. Let's imagine for a moment that Satoshi used 50 words that only Gavin and himself shared. Then let's say that Gavin shared 20 words only with Craig, 20 words only with Hal, and yet again 20 words only with Adam Back. The Doxer score would thus be 50 / ( (20 + 20 + 20) / 3 ) leaving us with a final Satoshi-Gavin score of 50 / 20. As you can see in this toy-example, Gavin shares more unique words with Satoshi than anyone else. We then proceed to repeat this process on every single one of the candidate texts and classify the highest score as Satoshi Nakamoto himself. 

Additionally, I've added a nifty feature with the -o switch that allows you to print out the words that the winner share with the unknown text. Of course when I conducted a one word gram (default setting) with the Bitcoin forum, I found that the winner Gavin Andresen shared an odd phrase of 'back-of-the-envelope' only with Satoshi. As you can see, Doxer leaves punctuation intact and tries to retain as much information as possible so as to find intricate results. 

And to overcome the problem of dispersion mentioned earlier, Doxer runs a quick Burrows' Delta to find the nearest neighbors of the query text. The list of top deltas can then be cut down to a predetermined amount. You may use the -r input with a specified number afterwards. For example `doxer.py -t 3satoshi -r 3` will cut the dataset of over 600 texts down to the nearest 3 texts so as to find more interesting unique words between these likely authors. Keep in mind that the Burrows Delta measure does not currently  include z-scores in the current program because I actually created a Random Forest to act as a reduce() function for the algorithm. Such was undertaken so as to get the best result possible. The amazing thing about the Random Forest is that it has the ability to reject all candidates so that Doxer's job doesn't have to deal with rubbish texts. I found this most useful when analyzing the Bitcoin whitepaper against around 50 other whitepapers. Every model of the Forest in fact rejected the other whitepapers, and thus I didn't have to waste time analyzing the closest neighbor. They were all rejected in one fell swoop! 

I took it upon myself to create my own feature collecting function by using skip grams so as to quicken up the pace. I devised a crafty little function to put gaps in the ngrams so that regardless of the number of grams I collect, the data is always represented as 4-grams, thus making the algorithm scalable to whatever number of grams I desire. For example, a frequent 4-gram set of characters are [t,h,e,n]. A frequent 2-gram of words may also be [of,the] or even [but,the]. My skip gram would reduce the gram [the,quick,brown,fox,jumped] down to [the,quick,fox,jumped] because I'm applying the skip gram pattern of [1,1,0,1,1] with the zero representing the 'brown' gram being dropped. Here is an example of how Doxer calculates the skip grams:  

```python
from doxer import Doxer

d = Doxer()

for y in range(1,20):
	print(d.split([0 for x in range(y)]))

[1]
[1, 1]
[1, 1, 1]
[1, 1, 1, 1]
[1, 1, 0, 1, 1]
[1, 0, 1, 0, 1, 1]
[1, 0, 1, 0, 0, 1, 1]
[1, 0, 1, 0, 0, 1, 0, 1]
[1, 0, 0, 1, 0, 0, 0, 1, 1]
[1, 0, 0, 1, 0, 0, 0, 1, 0, 1]
[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1]
[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
```

## General Usage

Find Satoshi in a folder of forum posts: 


	doxer.py -t 3satoshi -r 3 


Benchmark one of the folders up above with candidate reduction of 10: 

	doxer.py -b -r 10 

Find Satoshi using 4 character ngrams: 

	doxer.py -t 3satoshi -r 3 -c -n 4

Find Satoshi using 5 word ngrams: 

	doxer.py -t 3satoshi -r 3 -n 5


## Bootstrapping 

As already hinted at in the section above, stylometry has often suffered from the limitation of nearest neighbor analysis, having classified any old text in the absence of a truly worthy text to attribute authorship upon. The answer to this unfortunate problem is that of bootstrapping. 

And by way of example, let's say you conduct a nearest neighbor approach on a list of novels from one hundred years ago. You compare them all to Satoshi's forum posts and find that it returns the nearest neighbor being some obscure author that had nothing whatsoever to do with cryptocurrency. Can you see why we must employ something else in order to seal the deal and find Satoshi's identity once and for all? 

We will conduct our bootstrapping with what is called a Random Forest and these are the steps in which we'll take to extract the features: 

1. First we loop over all texts and create a list of top grams. We will only use the most frequent grams in the entire dataset. 

2. In this toy example published by Burrows, the unknown Milton text is being compared to a list of candidate texts, one being the famous Paradise Lost text. A distance measure is used to see how similar the two texts are in terms of gram usage. The Burrows Delta is merely an average of the absolute distances between frequencies. A z-score is made on each gram so as to not let one gram bias the entire average. We will not need z-scores in our Random Forest because the Forest does not average out the various distances.  

3. The closest Burrows Delta here indicates authorship. It is quite effective given its simplicity and ease of use. 

4. These are the absolute difference scores which I will use as features in the Forest. These features (i.e. attributes) will construct each instance in the pair wise comparison table. I will have around 200,000 of these instances for the Gutenberg corpus and around 80,000 for the Kindle corpus. A class column is appended on the end of each instance with a key of 1 = True and 0 = False. The classification therefore is a simple one of binary values. By the way, I will upload a `tar.gz` file of both the Gut and Kin datasets so that anyone can reconstruct the forest. I won't upload the actual pickles of the forests because it not good practice to encourage people to run serialized objects from the internet.  

5. One can see the top texts in this analysis being Paradise Lost and Paradise Regained. These texts are obviously very similar in style to that of the Milton text because term frequencies of their individual grams match closely to the text in question. 

I will use a Forest model built over these features so as to reduce the list of candidates down to a more reasonable number. And it is worth noting that the absence of z-scores means that the features are not altered by adding more texts to the corpus. The features will remain the same number whether you have only two texts or two million texts. 



![Burrows' Delta distance measure explained](images/burrows_deltas.png)


Source: 

https://doi.org/10.1093/llc/17.3.267

And by way of a surprise finding, I discovered that the Gutenberg corpus was awful in classifying the Satoshi texts. The Kindle dataset, being a much more recent collection of texts was much better at the task of reducing the candidates. I suspect that this is the case because the Gutenberg texts has older texts and the Kindle has much more newer texts. For this reason I was able to use Gut for reducing older say Biblical texts and Kin for the Satoshi problem. The Gut text has translated works such as Plato and Aristotle into English and so is well suited to Biblical texts. I used an anagram algorithm to discover any overlapping texts between Kin and Gut, thus removing them. For example, one dataset would have JaneAusten while the other would have AustenJane. A simple lowercase() and sorted() function can easily discover that these two texts are by the same author. And by the way, I suspect that some sort of purity measure, such as a gini coefficient, may prove useful and analyzing how useful a particular dataset is. For example, if a dataset is so awful that it classifies every author as the same then you can measure this and automatically select a better training set. This is just a thought for future research.  

I created 13 Forest models on an Amazon EC2 with character grams of 100 through to 1000 with gaps of 100 (10 models), and word grams of [100,200,300] (3 models). There were two words in the word grams that needed to be removed due to an error and so the 300 condition was really 298 (there was an empty string and a parsing error of 'ofthe' in the single word grams). I created 3 separate models for each condition being Kin, Gut, and a Combination of Gut/Kin. 

My findings during the reduce() stage was that two authors being Gavin Andresen (ID = 224) and Lachesis (ID = 237) were classified as Satoshi 9 times out of 13 for the Kin Random Forest models (3 word models and 10 4-gram character models). I then removed all texts under 4 classifications to complete the reduction process and to let Doxer analyze the texts with its unique word overlap algorithm. I analyzed character grams 4 through 10 (7 class) and word grams 1 through to 10 (10 class) on the Doxer algorithm, thus giving me 17 extra classifications. The results were quite definite as illustrated in the graph below with all but one classification going to Gavin. All character grams were given to Gavin and only the 10th word gram was given to Lachesis. Therefore, out of the top classified Random Forest candidates, Gavin Andresen had the most unique grams (character and word) with that of Satoshi Nakamoto! 

### Update 7/June/21
**I recently added Hal's posts and found that the Forest also added him to the reduced list. My top favorites of Satoshi now include Gavin, Hal, Lachesis (i.e. Eric Swanson). However, when I compare only Gavin, Hal, and Lachesis, Gavin wins.**

I further ran the reduced datasets over the rstylo library in R to see if I would get a similar result. Using 1,000 most frequent word models and also using both Forest texts above 4 classifications and all texts classified as Satoshi, I found that Gavin and Satoshi clustered together. It is interesting that even in the rstylo() cluster Lachesis was second best to Gavin, thus showing what a tight race it was indeed. 

Results of Doxer and RandomForest as to number of classifications as Satoshi: 

![Results of Doxer and Random Forest](images/forest-and-doxer-results.png)


Results of rstylo library clustering all the texts classified by Forest as Satoshi: 

![All texts classified by RandomForest clustered using rstylo](images/rstylo-large.png)


Results of rstylo library clustering only top texts above 4 Forest classifications: 

![Top texts over 4 classifications from Forest clustered using rstylo](images/rstylo-small.png)


And here is the rstylo R package: 

https://github.com/computationalstylistics/stylo

## Benchmark

As one of the more important parts of stylometry, a benchmark allows you to see at a first glance whether the algorithm has any utility to it. I took the time to benchmark three separate collections, each in a different language. I'm actually really impressed with this algorithm so as to get a fair result on three separate languages. 


English Novels: 83% (possibly 85% once Anon/Blackmore are taken into account)

Russian Novels: 85.18%

Polish Novels: 85%


This was the result for `doxer.py -b -r 10` 


The other wonderful thing about these results is that the Doxer algorithm could achieve this with no training sets. Yes, that's right, Doxer is an unsupervised algorithm making it quite light and easily customized. I also performed a benchmark on the PAN2014 dataset (pictured below) but didn't go into too much depth with it. That may be a project for a future date but doesn't concern me much here because the PAN2014 has quite small samples and doesn't resemble the Satoshi problem. I was able to beat many sub-sets on PAN2014 when using a combined training set and calibration model. Again, there is such ample room for customizing and extending such a simple algorithm as Doxer. Be my guest and pull it apart!

Here are the links to the original benchmarks: 

https://github.com/computationalstylistics/100_polish_novels

https://github.com/computationalstylistics/100_english_novels

https://github.com/JoannaBy/RussianNovels


## Interpretability `\w*-\w*-\w*`

I actually came across this regex pattern with the infamous `back-of-the-envelope` word cluster. I was reading over pages upon pages of Satoshi and then suddenly, when tossing through Gavin's blogs, I noticed this cluster of words in a blog published months before Gavin joined the Bitcoin project. But this pattern drips in fact with the style of Bitcoin writings. Satoshi's identity is embedded in the very term `proof-of-work` or `proof-of-stake`. It is as if Gavin's fingerprints themselves are all over this fine piece of machinery!

Here are some other regex results that contain the same pattern. Let's start with Satoshi: 

	one-person-one-vote
	proof-of-work
	tit-for-tat
	non-lower-ascii
	stable-with-respect-to-energy
	side-by-side


Now let's see the pattern used by Gavin: 

	fee-per-kilobyte
	some-amount-per-1000-bytes-of-the-transaction
	zero-knowledge-proofs
	three-to-one
	btc-to-fiat
	tragedy-of-the-commons
	business-as-usual

And of course the infamous: 

	back-of-the-envelope


Can you spot the similarity here? There is a certain style of writing that goes along with the use of such word clusters. But don't just observe the words by themselves, have a look at where they are placed. While musing over the -o switch in the Doxer program, I noticed that Gavin is distinguished from many of the other authors due to the added punctuation in his words. There are so many of these word clusters that are given added weight on Gavin because he places them at the end of a sentence with a fat full stop appended to the end. It is this added feature that lets me know that Doxer is not just analyzing unique words but is also analyzing the unique grammar of an author. Yes, many of the Bitcoin developers have used the same word clusters as proof-of-work, but they still haven't used them in the very unique way in which their creator used them. 

So next time you see one of those most common word clusters like proof-of-work, just remember that there is a greater regex pattern going on here that has Satoshi's fingerprints all over them. But don't take my word for it. Please be my guest and look through the texts yourself. I have so graciously compiled a very nice collection of texts in the folder above (as no research on this matter has done yet). I actually encourage you to do your own analysis. Extend the field of stylometry by all means. I would simply love it in fact if I could simply clone someone else's Github page and conduct cutting edge stylometry from the comfort of my own Linux terminal!

### Update 7/June/21

**Gavin also performs well on patterns of multiple words in a row. He not only uses the term back-of-the-envelope, but also uniquely adds the words "rough" or "my" at the front in the same way that Satoshi does. There are hundreds of patterns like these that distinguish Gavin from other Satoshi-candidates.**

**Try it for yourself... download the Bitcoin forum folder above and type the following in terminal:**

	grep -l "My back-of-the-envelope" * 
	
**And then try:**

	grep -l "rough back-of-the-envelope" * 
	
**Can you see the results? I'll print them out for you in case you haven't. Both commands in bash return the same result:**

	224GavinAndresen_forum.txt
	3satoshi_forum.txt
	
**That's right, out of 620 profiles from the BitcoinTalk Forum, only two people used these word combinations. That's 63.4 MB of text where only two profiles pop out as using this phrase. It just so happens that Stylometric analysis also isolates these two profiles as of having the same writing style. That's Stylometric analysis on function word use (i.e. frequency patterns of words like "the","but","then") and also unique word usage (i.e. Doxer).**

You can check out the use of this phrase in the wild on the BitcoinTalk forum:

Satoshi's use of the phrase:
https://bitcointalk.org/index.php?action=profile;u=3;sa=showPosts;start=460

Gavin's use of the phrase:
https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=200

And here are some screenshots of the phrase:

![Gavin's use of term "rough" and "my" back-of-the-envelope](images/gavin-satoshi-doxer.png)

![Gavin's use of "rough" and "my" back-of-the-envelope in wild next to Satoshi](images/gavin-satoshi-doxer2.png)

## Consensus

And by way of my own rather rough form of back-of-the-envelope analysis, I couldn't help but peer inside the mysteries of the Bible texts and see if I could solve a mystery or two. I particularly wanted to find out the author of the q-source gospel, the supposed original gospel of the new testament. In addition to this I wanted to test out the Septuagint priority theory that most of the old testament was written around or after the Septuagint translation. I reasoned to myself in a rather crude way that the priority would be supported if I found an earlier text of the Bible, let's say Genesis, loaded with a latter book such as Daniel or Tobit. What I found was quite impressive because it seems that even the simplest of models could handle this case. The book of Genesis has in fact a similar writing style to Kings I through to IV, but better yet, a salient similarity to that of Daniel! The R library rstylo in fact clusters Daniel and Genesis together even with all of the other LXX (Septuagint) books present. And if you read the books of Genesis and Daniel together you may notice that they have similar words and stories--both delve deeply into the world of dream interpretation, probably even a precursor or inspiration for the late Sigmund Freud. But have you ever noticed how similar Daniel is to Joseph (a character in Genesis). They both interpreted terrifying dreams of the king and were given a nice gold necklace to adorn their necks. I also noticed that with the Septuagint version, and surviving in texts such as the King James Bible, that both Daniel and Genesis used the word firmament, a possible indication as to the cosmology of the writer. 


Moving right along to the q-source gospel, I must say that I was deeply disappointed with my findings because it disproved a theory of mine that James the Just wrote the q-source. I had to face up to the facts when both the Forest and Doxer nearly completely ignored the James texts and instead, nearly in every single model, classified the Gospel of Thomas as the same author as the long lost q-source gospel. The finding was extremely robust and I had no choice but to accept the findings and drop James for probably ever more. In a rather interesting way, both testaments have a theoretical background in Greek philosophy. The possible Hellenistic origins of the old testament and the gnostic (i.e. middle Platonism) of Thomas are a clear indication that there is a Greek context to both old and new testaments. I chuckle to myself thinking about  my fundamentalist upbringing where Christmas, Easter, and Birthdays were all banned simply because they had origins in Greek and Roman festivities. To be quite frank, I cannot understand it for the life of me as to why anyone would hold such contempt for Greek culture and philosophy. If you were to ask me I think it is our birth right in the west to know of where our culture comes from and where we may be going as a result. Well I hope Doxer has solved all of these mysteries once and for all, but I'm quite sure a lot more work will need to be done on the matter. 


![Satoshi, Q-Source, and Septuagint](images/doxer-results.jpg)


Here are some interesting books that you can read on the matter while maybe also by chance running Doxer on the side:


The Gospel of Thomas and Plato

By Ivan Miroshnikov

https://brill.com/view/title/38096?language=en



Plato and the Creation of the Hebrew Bible

By Russell E. Gmirkin

https://www.routledge.com/Plato-and-the-Creation-of-the-Hebrew-Bible/Gmirkin/p/book/9780367878368



## Systematic Bias

One of the very reasons why I chose to analyze the Bitcoin forums is that it holds a rare chance in stylometry to perform what amounts to a controlled scientific experiment. I'm not sure whether stylometry could ever be one hundred percent scientific, but I regard the goal in of itself a worthy one to strive for. What I really love about the Bitcoin forums is that they have a large chunk of texts, volumes in fact, of people discussing matters in the same genre without any interruption. There is no, for example, deviation by way of writing fan fiction or publishing some long winded blog. Everything in the forum is tightly constrained to the topic of Bitcoin and cryptocurrency in general. For this reason a true stylometry measurement can be had by focusing on the differences only within this closed community. I hold this current research to be a valuable contribution to the field of stylometry for this reason and hold these results to be more pure then other textual comparisons that compare texts from a variety of sources. Even that of a blog site being compared to a forum is an example of a vastly different genre. 

When using the datasets I've provided, please remember that the author is often held at the front before the title of the book. However, I have changed it slightly so that the file Gorky-Mat_russian.txt means that the authorship text includes Gorky- and Mat as the algorithm splits at the `_` mark. If you don't have every author as a unique string then the program will crash. This is why this is important. 

The Forum corpus contains substantial texts between IDs 1 and 2000. I also included Adam Back simply because many people have suggested his authorship. When I say substantial I mean anything over one page of texts and with a valid ID. The filename 3satoshi_forum.txt means that the profile name is satoshi, while the ID is 3. One can access the raw data by visiting the html link as follows: 


https://bitcointalk.org/index.php?action=profile;u=3


The profile of Gavin Andresen is 224 which can be accessed here: 

https://bitcointalk.org/index.php?action=profile;u=224


As you can probably guess the u=3 and u=224 can be replaced with any ID that you may wish to look up so as to explore the actual texts in their original form. 

On a note about the validity of the Kin and Gut Forest models. They both had cross validity between one another at around 83% accuracy. Given that they have vastly different styles of texts being old and new, it is quite a testament to the Forest model that it is able to still find a high level of accuracy nevertheless. 

And on an extra note about the double space pattern associated with Adam Back. My simple response is that Gavin Andresen also had this pattern of two spaces after full stops in the early days of his forum posts. I don't know exactly what this pattern means but it seems many early forum posts had this pattern. Maybe it is a software feature? I have included a picture of the phenomena in Gavin's early posts down below.  

## Interpretability++ 



### back-of-the-envelope and double-spacing!

![Gavin, Satoshi, back-of-the-envelope, double-spacing](images/back-of-the-envelope-gavin-satoshi.jpg)



### non-obvious example of Satoshi
Other Bitcoin developers use this word, but in the reduced list, Gavin only uses it. 

![non-obvious example of Satoshi and Gavin](images/non-obvious-satoshi.jpg)



### double-spacing pattern 

![Double spacing, Satoshi, Gavin](images/double-spacing-satoshi-gavin.jpg)



### Blackmore and Clara finding


![Doxer finds Clara and Blackmore](images/clara-and-blackmore.jpg)



### Dream Interpretation 


![Joseph, Daniel, and Dream Interpretation](images/joseph-and-daniel.jpg)



### PAN2014 Benchmark Results (not complete yet)


![PAN2014 Benchmark Result](images/pan2014-benchmark.jpg)


# Working out

## Doxer Results after using word gram Forest reduction:
1. Note that nearly *all* of the results declare Gavin the winner.
2. Also note that on the word gram forests, both Gavin and Hal are classified 2/3 times.
3. The reduced list is from a folder of 600 profiles and reduces them down to a small number. 

```python
Running over all Forest models
--------------------------------------
56 1.0 ['2436Hal_forum', '3satoshi_forum']
56 1.0 ['2436Hal_forum', '3satoshi_forum']
59 1.0 ['1783Artefact2_forum', '3satoshi_forum']
144 1.0 ['224GavinAndresen_forum', '3satoshi_forum']
179 1.0 ['1171btchris_forum', '3satoshi_forum']
219 1.0 ['517BeeCee1_forum', '3satoshi_forum']
16 1.0 ['526Olipro_forum', '3satoshi_forum']
144 1.0 ['224GavinAndresen_forum', '3satoshi_forum']
276 1.0 ['597omegadraconis_forum', '3satoshi_forum']
301 1.0 ['511Bitquux_forum', '3satoshi_forum']
369 1.0 ['469eugene2k_forum', '3satoshi_forum']
372 1.0 ['1168puddinpop_forum', '3satoshi_forum']
434 1.0 ['3satoshi_forum', '357EricJ2190_forum']
498 1.0 ['3satoshi_forum', '466lfm_forum']
508 1.0 ['3satoshi_forum', '479PulsedMedia_forum']
578 1.0 ['3satoshi_forum', '1567doublec_forum']
609 1.0 ['3satoshi_forum', '576Syke_forum']

Reduced list of forest candidates:
--------------------------------------
[[17, '3satoshi_forum'], [2, '2436Hal_forum'], [2, '224GavinAndresen_forum'], [1, '597omegadraconis_forum'], [1, '576Syke_forum'], [1, '526Olipro_forum'], [1, '517BeeCee1_forum'], [1, '511Bitquux_forum'], [1, '479PulsedMedia_forum'], [1, '469eugene2k_forum']]

Analyzing Word Grams 1 - 10
--------------------------------------
1 224GavinAndresen
2 224GavinAndresen
3 224GavinAndresen
4 224GavinAndresen
5 224GavinAndresen
6 597omegadraconis
7 224GavinAndresen
8 224GavinAndresen
9 224GavinAndresen

Analyzing Char Grams 4 - 10
--------------------------------------
4 224GavinAndresen
5 224GavinAndresen
6 224GavinAndresen
7 224GavinAndresen
8 224GavinAndresen
9 224GavinAndresen
```

## Doxer results on character n-gram built forest reduction
1. There are more classifications because there are 10 different forest models. 
2. Gavin wins again overall, yet faces competition from Hal. 
3. It is of interest that Hal has such a similar writing style as that of Satoshi. 
4. Also worth noticing is that Lachesis wasn't found in the word gram models previous, but is found in the character gram models. 

```python
Running over all Forest models
--------------------------------------
14 1.0 ['413bg002h_forum', '3satoshi_forum']
33 1.0 ['392ribuck_forum', '3satoshi_forum']
56 1.0 ['2436Hal_forum', '3satoshi_forum']
87 1.0 ['1929davout_forum', '3satoshi_forum']
214 1.0 ['541jgarzik_forum', '3satoshi_forum']
242 1.0 ['1565barbarousrelic_forum', '3satoshi_forum']
281 1.0 ['525Red_forum', '3satoshi_forum']
393 1.0 ['237lachesis_forum', '3satoshi_forum']
450 1.0 ['3satoshi_forum', '535em3rgentOrdr_forum']
490 1.0 ['3satoshi_forum', '1496Bimmerhead_forum']
501 1.0 ['3satoshi_forum', '491kiba_forum']
572 1.0 ['3satoshi_forum', '14The Madhatter_forum']
599 1.0 ['3satoshi_forum', '704caveden_forum']
33 1.0 ['392ribuck_forum', '3satoshi_forum']
56 1.0 ['2436Hal_forum', '3satoshi_forum']
87 1.0 ['1929davout_forum', '3satoshi_forum']
144 1.0 ['224GavinAndresen_forum', '3satoshi_forum']
242 1.0 ['1565barbarousrelic_forum', '3satoshi_forum']
281 1.0 ['525Red_forum', '3satoshi_forum']
393 1.0 ['237lachesis_forum', '3satoshi_forum']
572 1.0 ['3satoshi_forum', '14The Madhatter_forum']
33 1.0 ['392ribuck_forum', '3satoshi_forum']
87 1.0 ['1929davout_forum', '3satoshi_forum']
144 1.0 ['224GavinAndresen_forum', '3satoshi_forum']
281 1.0 ['525Red_forum', '3satoshi_forum']
393 1.0 ['237lachesis_forum', '3satoshi_forum']
599 1.0 ['3satoshi_forum', '704caveden_forum']
14 1.0 ['413bg002h_forum', '3satoshi_forum']
33 1.0 ['392ribuck_forum', '3satoshi_forum']
56 1.0 ['2436Hal_forum', '3satoshi_forum']
87 1.0 ['1929davout_forum', '3satoshi_forum']
144 1.0 ['224GavinAndresen_forum', '3satoshi_forum']
214 1.0 ['541jgarzik_forum', '3satoshi_forum']
242 1.0 ['1565barbarousrelic_forum', '3satoshi_forum']
364 1.0 ['198allinvain_forum', '3satoshi_forum']
393 1.0 ['237lachesis_forum', '3satoshi_forum']
498 1.0 ['3satoshi_forum', '466lfm_forum']
56 1.0 ['2436Hal_forum', '3satoshi_forum']
87 1.0 ['1929davout_forum', '3satoshi_forum']
144 1.0 ['224GavinAndresen_forum', '3satoshi_forum']
219 1.0 ['517BeeCee1_forum', '3satoshi_forum']
242 1.0 ['1565barbarousrelic_forum', '3satoshi_forum']
599 1.0 ['3satoshi_forum', '704caveden_forum']
33 1.0 ['392ribuck_forum', '3satoshi_forum']
56 1.0 ['2436Hal_forum', '3satoshi_forum']
87 1.0 ['1929davout_forum', '3satoshi_forum']
242 1.0 ['1565barbarousrelic_forum', '3satoshi_forum']
364 1.0 ['198allinvain_forum', '3satoshi_forum']
393 1.0 ['237lachesis_forum', '3satoshi_forum']
599 1.0 ['3satoshi_forum', '704caveden_forum']
33 1.0 ['392ribuck_forum', '3satoshi_forum']
41 1.0 ['4sirius_forum', '3satoshi_forum']
56 1.0 ['2436Hal_forum', '3satoshi_forum']
87 1.0 ['1929davout_forum', '3satoshi_forum']
242 1.0 ['1565barbarousrelic_forum', '3satoshi_forum']
393 1.0 ['237lachesis_forum', '3satoshi_forum']
476 1.0 ['3satoshi_forum', '336Insti_forum']
14 1.0 ['413bg002h_forum', '3satoshi_forum']
55 1.0 ['270llama_forum', '3satoshi_forum']
56 1.0 ['2436Hal_forum', '3satoshi_forum']
144 1.0 ['224GavinAndresen_forum', '3satoshi_forum']
393 1.0 ['237lachesis_forum', '3satoshi_forum']
56 1.0 ['2436Hal_forum', '3satoshi_forum']
144 1.0 ['224GavinAndresen_forum', '3satoshi_forum']
393 1.0 ['237lachesis_forum', '3satoshi_forum']
56 1.0 ['2436Hal_forum', '3satoshi_forum']
144 1.0 ['224GavinAndresen_forum', '3satoshi_forum']
393 1.0 ['237lachesis_forum', '3satoshi_forum']

Reduced list of forest candidates:
--------------------------------------

[[68, '3satoshi_forum'], [9, '2436Hal_forum'], [9, '237lachesis_forum'], [7, '224GavinAndresen_forum'], [7, '1929davout_forum'], [6, '392ribuck_forum'], [6, '1565barbarousrelic_forum'], [4, '704caveden_forum'], [3, '525Red_forum'], [3, '413bg002h_forum']]

Analyzing Word Grams 1 - 10
--------------------------------------
1 224GavinAndresen
2 224GavinAndresen
3 224GavinAndresen
4 237lachesis
5 2436Hal
6 2436Hal
7 2436Hal
8 2436Hal
9 2436Hal

Analyzing Char Grams 4 - 10
--------------------------------------
4 224GavinAndresen
5 224GavinAndresen
6 224GavinAndresen
7 224GavinAndresen
8 224GavinAndresen
9 224GavinAndresen
```

## Doxer results just comparing Satoshi, Gavin, Hal, and Lachesis
1. This may seem repetitive, but it is worth noting that when the top contenders are compared with no other texts, that Gavin is more similar to Satoshi. 

```python
Analyzing Word Grams 1 - 10
--------------------------------------
1 224GavinAndresen
2 224GavinAndresen
3 224GavinAndresen
4 224GavinAndresen
5 224GavinAndresen
6 224GavinAndresen
7 224GavinAndresen
8 224GavinAndresen
9 224GavinAndresen

Analyzing Char Grams 4 - 10
--------------------------------------
4 224GavinAndresen
5 224GavinAndresen
6 224GavinAndresen
7 224GavinAndresen
8 224GavinAndresen
9 224GavinAndresen
```
# For those still not convinced

I used the library `faststylometry` to see who is the closest neighbor of Satoshi out of the 600+ profiles in the above folder. 

1) install `faststylometry`:

https://github.com/fastdatascience/faststylometry

```pip install faststylometry```

2) rename files to adhere to faststylometry program: 

```
101601adam3us_-_forum.txt
101Goldstein_-_forum.txt
1034gumtree_-_forum.txt
1036maxinedougherty_-_forum.txt
1045torservers_-_forum.txt
1047Loki_-_forum.txt
1048pavelo_-_forum.txt
1050Guybrush_-_forum.txt
1052snrlx_-_forum.txt
1059wirher_-_forum.txt
```
3) run the code to find out closest neighbor to Satoshi using Burrows' Delta: 
```python
from faststylometry import Corpus
from faststylometry import load_corpus_from_folder
from faststylometry import tokenise_remove_pronouns_en
from faststylometry import calculate_burrows_delta
from faststylometry import predict_proba, calibrate
import nltk ; nltk.download("punkt")
train_corpus = load_corpus_from_folder("/home/flak/Documents/prog/.git/tests/train/")
train_corpus.tokenise(tokenise_remove_pronouns_en)
test_corpus = load_corpus_from_folder("/home/flak/Documents/prog/.git/tests/test/", pattern=None)
test_corpus.tokenise(tokenise_remove_pronouns_en)
a = calculate_burrows_delta(train_corpus, test_corpus, vocab_size = 300)
from operator import itemgetter ; first_item = itemgetter(0)
y = [[x[1][0],x[0]] for x in zip(a.index.values.tolist(),a.values.tolist()) if x[1][0] > 0] ; print(sorted(y, key=first_item))
```

4) here are the results with Gavin at the top: 

```
[[0.16304531949753703, '224GavinAndresen'],
[0.16819089987038993, '2436Hal'],
[0.17117725119873753, '237lachesis'],
[0.18689911496747535, '1171btchris'],
[0.18902020407080056, '392ribuck'],
[0.1905760032509449, '601nelisky'],
[0.191341605504933, '143laszlo'],
[0.19470326629850976, '277jago2598'],
[0.19787557885928606, '1931genjix'],
[0.19803556342323256, '466lfm'],
[0.20013588602407642, '49Cdecker'],
[0.20504875859359808, '541jgarzik'],
[0.2053947464440863, '413bg002h'],
[0.20603693932506034, '381FreeMoney'],
[0.207102693502892, '14The Madhatter'],
[0.20827964686736655, '490ByteCoin'],
[0.20833283606966607, '525Red'],
[0.2090643885444086, '1567doublec'],
[0.20995692671711028, '1998bober182'],
[0.21291032880876237, '1496Bimmerhead'],
[0.21363070576744642, '704caveden'],
[0.21394524686598132, '4sirius'],
[0.2150249002955321, '1882da2ce7'],
[0.21561899268622292, '1268nanotube'],
[0.21562366403733194, '1652harding'],
[0.21567581212773088, '26NewLibertyStandard'],
[0.21577919912647062, '336Insti'],
[0.21602612863249646, '345knightmb'],
[0.21622382313104055, '325jimbobway'],
[0.2163952408346549, '1864Drifter'],
[0.21656403021510498, '39fergalish'],
[0.21794367848910742, '430aceat64'],
[0.2192537090855312, '545melvster'],
[0.22094621987398672, '163Karmicads'],
[0.2213116420747501, '310martin'],
[0.22217650240609293, '517BeeCee1'],
[0.22286648170864387, '13SmokeTooMuch'],
[0.22293531802625202, '694BrightAnarchist'],
[0.2231266182870886, '643MoonShadow'],
[0.22331756645680917, '491kiba'],
[0.2233369431480121, '1775ShadowOfHarbringer'],
...
]
```
And so on, I cut the list off here because you get the idea. 

## Scree Plot Analysis of Basic Nearest Neighbor 


![Scree Plot of Gavin and Satoshi](images/scree-plot-gavin-satoshi-2.png)

## Out of top contenders (Gavin,Hal,Lachesis) who has noticeable similarity to Satoshi? 

I just noticed that Gavin used the term `proof-of-work` and that Hal and Lachesis **never** used this term in their forum posts! But didn't Hal invent proof of work? Yes he did and called it `proof of work (POW)` without using dashes to separate the words. He didn't use dashes even on the btc forum posts because it was his invention and didn't write it like that. Lachesis never uses the term `proof-of-work` in all of his forum posts. This is one of the many reasons why I settle on Gavin and Gavin alone as Satoshi. 

![proo-of-work versus proof of work with Satoshi and Gavin](images/proof-of-work-gavin-satoshi.png)

## Higher Priority Things ⭐
**This is a very unique phrase that should make people check twice**
1. "higher priority things" is listed on google.com.au as About 14,300 results (0.35 seconds). This is in contrast to "back-of-the-envelope" which is at About 5,080,000 results (0.38 seconds). 
2. Keep in mind that there are hundreds of unique terms that isolate Gavin as Satoshi, but "higher priority things" is a powerful indicator of authorship just by itself. 
3. When I run ```grep -l "higher priority things"``` over the entire 600+ pofiles, only Gavin and Satoshi use it. Even when I run grep on a collection of texts including Wei Dai, Nick Szabo, and Craig Wright, only Gavin and Satoshi use it!

### Satoshi wrote the following:
```It is possible, but it's a lot of work, and there are a lot of other``` **higher priority things** ```to work on.```

### Gavin wrote the following:
```I still plan on writing up why I disagree with the idea that a larger block size will lead to centralization, but I'm working on some``` **higher priority things** ```first.```

```We've got a VERY VERY long way to go before bitcoins are as popular as dollars, and there are much``` **higher priority things** ```to work on right now than adding more divisibility for a problem that is pretty likely to never actually be a problem.```

```I'm tempted to code that up and run some tests on a testnet-in-a-box, but there are much``` **higher priority things** ```on my TODO list;```

```So I'd really like to see network and client support for having both people pre-sign and hold on to a transaction with a far-in-the-future lockTime (maybe as a fee-only transaction). Nope, sorry, have``` **higher priority things** ```to do.```

```It is not on my short-term TODO list because there are too many other``` **higher priority things** ```on my TODO list, but a nice clean well-tested upward-compatible patch would be most welcome.```

```Something like this is possible (I've been thinking about doing it, although I have``` **higher priority things** ```on my TODO list)```

```I think that is a good idea (I think people would find all sorts of interesting uses for it), but there are``` **higher priority things** ```on the development roadmap.```

### Gavin uses phrase 'higher-priority things' in 2008

```We're irrationally anchored to the things that we already have, and that makes it really hard to give them up even though there might be other,``` **higher-priority things** ```we could (and should!) invest in. I'll keep reading Predictably Irrational; I'm hoping Professor Ariely will give some strategies for overcoming our irrational tendencies.```

https://gavinthink.blogspot.com/2008/02/pool-anchors.html

He suggests that he got the idea from reading **Predictably Irrational**, but I found the term ```higher priority``` in a nother book Gavin read called **The Bottom Billion.** Again, Gavin was using 'higher-priority things' in Feb 2008 way before 'meeting' Satoshi. The same thing is observed with phrase 'back-of-the-envelope' with both using it before meeting. Satoshi used the phrase 'higher priority things' in May 18 2010, this is a month before Gavin and Satoshi met in June 12 2010!

https://bitcointalk.org/index.php?action=profile;u=3;sa=showPosts;start=420

## A menace to the network

### Satoshi wrote:
```So much of the design depends on all nodes getting exactly identical results in lockstep that a second implementation would be``` **a menace to the network.**

### Gavin wrote:
```They'll either hack the existing code or write their own version, and will be``` **a menace to the network.**

## Don't have time to 
**This is actually a part of a famous Satoshi saying that most people would remember**

### Satoshi wrote:
```If you don't believe me or don't get it, I``` **don't have time to** ```try to convince you, sorry.```

**(This was written to Larimer who is under the name bytemaster [ID = 611], also available in the folder above).**

### Gavin wrote:
```I am also sorry I didn't speak up about some of the things that were said about you earlier in this thread (e.g. suggesting an attack on your pool is NOT cool) but I``` **don't have time to** ```read every forum post as soon as it is posted...```

## Probably be something like
**This one is kind of ironic because I'm detecting Satoshi using probability coefficients (term frequencies have been termed probabilities in the past).**

### Satoshi wrote:
```If you still have communication with the rest of your area, it would``` **probably be something like** ```1/1000 of the world or less.```

### Gavin wrote:
```The agenda will probably be decided ten minutes before the meeting, and will``` **probably be something like** ```"Peter talks for ten minutes and answers questions for 20 minutes while everybody is eating.```

## If I recall correctly

### Satoshi wrote:
**If I recall correctly,** ```500 is the prescribed status code for JSON-RPC error responses.```

### Gavin wrote:
**If I recall correctly,** ```300GB per month was the limit for my ISP in Australia, too.```

**If I recall correctly,** ```if they DO get mined into a block by somebody then they are displayed.```

**If I recall correctly,** ```the RPC importprivkey should be the only place where the normal memory allocator is used (the keys exist as ordinary hex strings in memory before they are processed by the importprivkey code).```

**If I recall correctly,** ```it does log the time at startup; for this particular test, I'm mostly interested in how long the upgrade process takes, so the old version not supporting timestamps isn't a problem.```

**If I recall correctly,** ```one of the libraries it links with is incompatible with 10.5.```

**If I recall correctly,** ```mybitcoin.com users got 49% of their coins back (I was one of the stupid people who trusted some bitcoins to them, by the way).```

**If I recall correctly,** ```the first testnet-in-a-box node should have all the coins.```

**If I recall correctly,** ```CO2 concentrations were pretty darn high when the dinosaurs where walking around.```

**If I recall correctly,** ```the courts have ruled that "commercial speech" is not as protected-- so laws that restrict (for example) cigarette ads on television are OK.```

**If I recall correctly,** ```people were saying exactly the same thing about URLs 10 years ago (...google... yup).```

**If I recall correctly,** ```getting around one-per-IP-address and CAPTCHA restrictions costs a scammer a few US pennies.```

**If I recall correctly,** ```people were very creative about finding ways to get around it, but I don't have any references handy.```

**If I recall correctly** ```(and I probably don't), the percentage of nodes currently on the network that are accepting incoming connections and the -maxconnections limit isn't great enough to support every node trying to make 25 outbound connections.```

**If I recall correctly,** ```after he was done he found that it wouldn't compile on Windows any more.```

**If I recall correctly,** ```there is a very small chance if you lose power or bitcoin crashes a key from the keypool could be used twice.```

## It Doesn't look to me like

### Satoshi wrote:
**It doesn't look to me like** ```Crypto++ could be deciding whether to use SSE2 at runtime.```

### Gavin wrote:
```For low-priority transactions,``` **it doesn't look to me like** ```many miners are accepting lower fees.```

## is a tiny little

### Satoshi wrote:
```mingwm10.dll``` **is a tiny little** ```DLL that came with the MinGW compiler that you need when you build for multi-thread.```

### Gavin wrote:
```If it``` **is a tiny little** ```niche thing then it is much easier for politicians or banks to smother it, paint it as "criminal money", etc. ```

## that in 20 years
**I like this one. Keep in mind that when multiple patterns like these are combined they can be used by doxer to make authorship attribution. There are lots and lots of boring overlaps in addition to these idiosyncratic ones.**

### Satoshi wrote:
```I'm sure``` **that in 20 years** ```there will either be very large transaction volume or no volume.```

### Gavin wrote:
```I think it is very unlikely``` **that in 20 years** ```we will need to support more Bitcoin transactions than all of the cash, credit card and international wire transactions that happen in the world today (and that is the scale of transactions that a pretty-good year-2035 home computer and network connection should be able to support).```

```I can imagine``` **that in 20 years** ```the US still has the world's biggest army but some other nation has the world's largest economy.```


# The origins of Gavin/Satoshi use of "back-of-the-envelope"

## May 23, 2008

### Gavin reads 'Power to save the world' by Gwyneth Cravens

```I've just finished reading "Power to Save the World," which is all about why using uranium to generate power is a really good idea.```

https://gavinthink.blogspot.com/2008/05/radiation-monster.html

Here is that very book:

https://www.amazon.com/Power-Save-World-Nuclear-Energy/dp/0307385876

And here is a quote from ```Power to save the world```:

```Rip gathered a few Seabed veterans, including Leo Gómez, and they pored over the Westinghouse data, did a``` **back-of-the-envelope** ```performance assessment, and after a couple of hours they concluded that, if the data was accurate, WIPP was going to fail because of a brine pocket in the salt bed caused by a one-liter-per-day leak. The group wrote a paper saying that, based on the information at hand, the repository would fill with water in a few years and the salt bed would eventually wash away. Gómez noted that the projected release of radionuclides would have been "enough to cook a cow."```

## November 18, 2008 (Gavin)

### Gavin uses the phrase back-of-the-envelope himself

```So I did a``` **back-of-the-envelope** ```calculation to see how much sprawl it might cause, worst case. Take the 10,000 or so households in Amherst, count each as a "dwelling unit", multiple by 1,000 square feet, and you get: about 230 acres. Which is just a little over 1% of the total acreage in Amherst.```

https://gavinthink.blogspot.com/2008/11/fall-town-meeting.html

##  February 23, 2010 (Satoshi)

### Satoshi uses the phrase before the date he was said to have met Gavin

**My back-of-the-envelope** ```projection: 42032 blocks/2016 = 20.85 = 85% of the way.  About 1.5 days to go until the next one.  That'll only be about 10 days since the last one, the target is 14 days, so 14/10 = 1.4 = around 40% difficulty increase.```

https://bitcointalk.org/index.php?action=profile;u=3;sa=showPosts;start=460

## June 12, 2010

Satoshi and Gavin supposedly make contact for the first time (This means that they both used this phrase before coming into contact). 

## July 17, 2010, (Satoshi)

```If one has a slight head start, it'll geometrically spread through the network faster and get most of the nodes.``` **A rough back-of-the-envelope** ```example:```

https://bitcointalk.org/index.php?action=profile;u=3;sa=showPosts;start=300

## November 09, 2010 (Gavin)

**Back-of-the-envelope:** ```Lets say computers in a few years can do a quadrillion hashes per second-- that's about 2^50 hashes/second.```

https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=1960

## February 02, 2011 (Gavin)

```Let me see if I can do a``` ***back-of-the-envelope:***

https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=1740

## February 10, 2011 (Gavin)

```If nobody beats me to it, I'll try to do a``` **back-of-the-envelope** ```calculation later today.```

https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=1720

## February 10, 2011 (Gavin)

```Here's the``` **back-of-the-envelope** ```calculation I used to get to that number:Transaction size:  ~300 bytes```

https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=1700

## February 11, 2011 (Gavin)

```There ARE hidden costs; the reason I want to do the``` **back-of-the-envelope** ```is to figure out how big those hidden costs are now and how big they're likely to get in the future.```

https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=1700

## March 14, 2011 (Gavin)

```1. OP_CHECKSIG drives network-wide costs (see the thread on network-wide transaction cost``` **back-of-the-envelope** ```calculations).```

https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=1540

## March 07, 2013 (Gavin)

```I did send them a pointer to this very``` **rough back-of-the-envelope** ```estimate on the current marginal cost of transactions:  https://gist.github.com/gavinandresen/5044482(if anybody wants to do a better analysis, I'd love to read it).```

https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=280

## March 10, 2013 (Gavin)

```Ok, fine, so do a``` **back-of-the-envelope** ```for what THAT cost is.```

**Rough, back-of-the-envelope:** ```how much does it cost to keep a dust-like transaction output in the unspent outputs set for 20 years?```

https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=260

## April 28, 2013 (Gavin)

**My back-of-the-envelope** ```calculations say that anybody willing to spend a few hundred dollars a year on a dedicated server with a high-bandwidth connection can support a MUCH, MUCH larger block size.The block size will be raised. Your video will just make a lot of people worried about nothing, in exactly the same way Luke-Jr's BIP17 proposal last year (and his hyperbolic rhetoric about BIP16) did nothing but cause a tempest in a teapot.```

https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=200

## October 6, 2014 (Gavin)

```Please don't be lazy, at least do some``` **back-of-the-envelope** ```calculations to justify your numbers (to save you some work: the average Bitcoin transaction is about 250 bytes big).```

https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=60

## July 31, 2020 (Gavin)

### Gavin still uses this phrase back-of-the-envelope even in 2020

```But maybe they're not insane; maybe the exercise and mental health benefits outweigh the risk. A few days ago I stumbled across some tools that let me do a rough``` **back-of-the-envelope** ```on the size of the risks, and I think we should encourage a lot more physically-distanced outdoor dancing (and singing and yoga and drumming and whatever else makes people happy).```

https://gavinthink.blogspot.com/2020/07/dancing-outside.html

