################################################################
# Author: Joshua Fantillo
# File: get_data_frame.py
# Date: April 11, 2022
# Employer: The University of the Fraser Valley
# Location: Abbotsford, BC, Canada
# Description: This file gets the features from get_attributes.py
# and builds a dataframe from those features. This file can be 
# modified to include and remove different features. It then 
# saves the dataframe to a csv file. This file gets the features
# from the gap-copurs from convokit.
################################################################

import get_attributes
import get_gap_corpus
import pandas as pd
from convokit import TextParser
from convokit import PolitenessStrategies

#gets the group number from the utternace 
def group(utt):
	group = utt.split(".")
	return group[0]

#gets each groups sentence and saves it into an array
def group_sentences(group_number, corpus, utter):
	group_sentences = []	
	for item in utter:
		if int(group(item)) == group_number:
			group_sentences.append(corpus.get_utterance(item).text)
	return group_sentences

#Gets the first utterance_id in each group	
def get_first_utter(corpus, utter):
	first_utter = []
	check_array = []
	for item in utter:
		if int(group(item)) not in check_array:
			first_utter.append(item)
			check_array.append(int(group(item)))	
	j = 1
	for i in range(len(first_utter)):
		j = i+1
		while j < len(first_utter):
			if int(group(first_utter[i])) > int(group(first_utter[j])):
				first_utter[j], first_utter[i] = first_utter[i], first_utter[j]
			j += 1
	return first_utter
	
# Gets all the data from the get_attributes.py file	
def get_group_data():
	data_frame = pd.DataFrame()
	corpus, utter = get_gap_corpus.gap()
	first_utter = get_first_utter(corpus, utter)
	ps = PolitenessStrategies()
	parser = TextParser(verbosity=1000)
	polite_corpus = parser.transform(corpus)
	polite_corpus = ps.transform(polite_corpus, markers=True)
	tok = []
	tot_sent = []
	tot_word = []
	avg_word = []
	sent = []
	filled = []
	coughs = []
	rand_noise = []
	laugh = []
	unc_word = []
	cos_sim = []
	unq_word = []
	word_freq = []
	bag_word = []
	bag_tag = []
	unq_tag = []
	tag_freq = []
	tok_rat = []
	tag_rat = []
	met_size = []
	met_len = []
	ags = []
	g_te = []
	g_ww = []
	g_tm = []
	g_eff = []
	g_qw = []
	g_sat = []
	poli = []
	
	group = 1
	
	while group < 29:
		tokens = get_attributes.tokens(group_sentences(group, corpus, utter))
		tok.append(tokens)
		poli.append(get_attributes.get_politeness(polite_corpus, utter, group))
		tot_sent.append(get_attributes.total_sentences(tokens))
		tot_word.append(get_attributes.total_words(tokens))
		avg_word.append(get_attributes.AWPS(tokens))
		sent.append(get_attributes.sentiment(group, corpus, utter))
		filled.append(get_attributes.filled_pauses(tokens))
		coughs.append(get_attributes.cough(tokens))
		rand_noise.append(get_attributes.random_noise(tokens))
		laugh.append(get_attributes.laughs(tokens))
		unc_word.append(get_attributes.unclear_words(tokens))
		tokens = get_attributes.clean_token(tokens)
		cos_sim.append(get_attributes.cosine_sim(tokens))
		total_unique_words, words_with_freq = get_attributes.word_freq(tokens)
		unq_word.append(total_unique_words)
		word_freq.append(words_with_freq)
		words_and_tags = get_attributes.pos_tags(tokens)
		bag_of_words, bag_of_tags = get_attributes.bag_of_words_and_tags(words_and_tags)
		bag_word.append(bag_of_words)
		bag_tag.append(bag_of_tags)
		total_unique_tags, tags_with_freq = get_attributes.word_freq(bag_of_tags)
		unq_tag.append(total_unique_tags)
		tag_freq.append(tags_with_freq)
		tok_rat.append(get_attributes.type_ratio(total_unique_words, bag_of_words))
		tag_rat.append(get_attributes.type_ratio(total_unique_tags, bag_of_tags))
		met_size.append(get_attributes.meeting_size(corpus, first_utter, group))
		met_len.append(get_attributes.meeting_length(corpus, first_utter, group))
		ags.append(get_attributes.AGS(corpus, first_utter, group))
		g_te.append(get_attributes.group_TE(corpus, first_utter, group))
		g_ww.append(get_attributes.group_WW(corpus, first_utter, group))
		g_tm.append(get_attributes.group_TM(corpus, first_utter, group))
		g_eff.append(get_attributes.group_Eff(corpus, first_utter, group))
		g_qw.append(get_attributes.group_QW(corpus, first_utter, group))
		g_sat.append(get_attributes.group_Sat(corpus, first_utter, group))
		group += 1
	data_frame = build(data_frame, tok, tot_sent, tot_word, avg_word, sent, filled, coughs, 				rand_noise, laugh, unc_word, cos_sim, unq_word, word_freq, bag_word, 			bag_tag, unq_tag, tag_freq, tok_rat, tag_rat, met_size, met_len, ags, 			g_te, g_ww, g_tm, g_eff, g_qw, g_sat, poli)
	return data_frame
	
# Builds the dataframe using different features that you can choose to include or disclude
def build(data_frame, tok, tot_sent, tot_word, avg_word, sent, filled, coughs, rand_noise, 			laughs, unc_words, cos_sim, unq_word, word_freq, bag_word, bag_tag, unq_tag, 		tag_freq, tok_rat, tag_rat, met_size, met_len, ags, g_te, g_ww, g_tm, g_eff, 		g_qw, g_sat, poli):
	#data_frame["Tokens"] = tok
	#data_frame["Total Sentences"] = tot_sent
	#data_frame["Total Words"] = tot_word
	data_frame["Avg Words Per Sentence"] = avg_word
	data_frame["Sentiment"] = sent
	#data_frame["Filled Pauses"] = filled
	#data_frame["Coughs"] = coughs
	#data_frame["Random Noise"] = rand_noise
	#data_frame["Laughs"] = laughs
	#data_frame["Unclear Words"] = unc_words
	data_frame["Cosine Similarity"] = cos_sim
	#data_frame["Unique Words"] = unq_word
	#data_frame["Word Frequency"] = word_freq
	#data_frame["Bag of Words"] = bag_word
	#data_frame["Bag of Tags"] = bag_tag
	#data_frame["Unique Tags"] = unq_tag
	#data_frame["Tag Frequency"] = tag_freq
	#data_frame["Token Ratio"] = tok_rat
	#data_frame["Tag Ratio"] = tag_rat
	#data_frame["Meeting Size"] = met_size 
	data_frame["Meeting Length in Minutes"] = met_len
	data_frame["AGS"] = ags
	#data_frame["Group TE"] = g_te
	#data_frame["Group WW"] = g_ww 
	#data_frame["Group TM"] = g_tm
	data_frame["Group Eff"] = g_eff	
	data_frame["Group QW"] = g_qw
	#data_frame["Group Sat"] = g_sat
	data_frame["Politeness"] = poli
	#print(data_frame)
	return data_frame
	
# saves the dataframe to a csv
def save_data_frame(data_frame):
	data_frame.to_csv('research_data')
	
def main():
	save_data_frame(get_group_data())
	
if __name__ == "__main__":
	main()
	

	
