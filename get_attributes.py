################################################################
# Author: Joshua Fantillo
# File: get_attributes.py
# Date: April 11, 2022
# Employer: The University of the Fraser Valley
# Location: Abbotsford, BC, Canada
# Description: This program gets many different 
# features from the GAP corpus Dataset. This file
# is meant to be accessed through the get_data_frame.py file.
################################################################

import nltk
import convokit
import get_data_frame
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.probability import FreqDist

# gets the groups tokens
def tokens(group_sentence):
	parsed_sentence = []
	for item in group_sentence: 
		item = item.lower()
		parsed_sentence.append(nltk.word_tokenize(item)[1:-1])
	return parsed_sentence					

# gets the groups total sentences
def total_sentences(token_sentence):
	total_sentences = 0
	for item in token_sentence:
		if item[0] != '$':
			total_sentences += 1
	return total_sentences
	
# gets the groups total words
def total_words(token_sentence):
	total_words = 0
	for item in token_sentence:
		for word in item:
			if word != "$":
				total_words += 1
	return total_words

# gets the groups average words per sentence
def AWPS(token_sentence):
	return total_words(token_sentence)/total_sentences(token_sentence)
	
# gets the average sentiment score for the group
def sentiment(group, corpus, utt_id):
	sentiment = []
	for item in utt_id:
		if int(get_data_frame.group(item)) == group:
			sentiment.append(corpus.get_utterance(item).meta['Sentiment'])
	average_sentiment = group_sentiment(sentiment)
	return average_sentiment

# returns the average positive or negative sentiment score for the group
def group_sentiment(sentiment):
	positive = 0
	negative = 0
	for item in sentiment:
		if item == 'Positive':
			positive += 1
		if item == 'Negative':
			negative += 1
	total = positive + negative
	if negative > positive:
		return -1 * negative/total
	else:
		if negative < positive:
			return positive/total
		else:
			return 0
			
# gets the number of filled pauses for the group		
def filled_pauses(token_sentence):
	filled_pauses = 0
	for item in token_sentence:
		for word in item:
			if word == 'uh' or word == 'oh':
				filled_pauses += 1
	return filled_pauses 
	
# gets the number of laughs in the group
def laughs(token_sentence):
	laugh = 0
	for item in token_sentence:
		for word in item:
			if word == '$':
				laugh += 1
	return laugh

# gets the number of random noises for each group	
def random_noise(token_sentence):
	noise = 0
	for item in token_sentence:
		for word in item:
			if word == '#':
				noise += 1
	return noise

#gets the number of coughs for each group
def cough(token_sentence):
	cough = 0
	for item in token_sentence:
		for word in item:
			if word == '%':
				cough += 1
	return cough
	
# gets the number of unclear words for the group
def unclear_words(token_sentence):
	unclear = 0
	check = False
	for item in token_sentence:
		for word in item:
			if check == True:
				if word == 'unclear':
					unclear += 1
				check = False		
			if word == '[':
				check = True
	return unclear

# cleans the token sentences	
def drop_sounds(token_sentence):
	new_tokens = []
	for item in token_sentence:
		if len(item) != 1:
			new_tokens.append(item)
		else:
			if item[0] != '$' and item[0] != '#' and item[0] != '%':
				new_tokens.append(item)
	return new_tokens	
	
# gets the stopwords for the tokenized sentences 
def stop_words(token_sentence):
	#nltk.download('stopwords')
	stop_words=set(stopwords.words("english"))
	hold = []
	new_token_sentence = []
	for i in range(len(token_sentence)):
		for word in token_sentence[i]:
			if word not in stop_words:
				hold.append(word)
		token_sentence[i] = hold
		hold = []
		return token_sentence

# lemmatizes the tokenized sentences
def lemmatize_sentence(token_sentence):
	#nltk.download('wordnet')
	#nltk.download('omw-1.4')
	lem = WordNetLemmatizer()
	for sentence in range(len(token_sentence)):
		for word in range(len(token_sentence[sentence])):
			 token_sentence[sentence][word] = lem.lemmatize(token_sentence[sentence][word],"v")		
	return token_sentence

# gets the average cosine similarity for the groups sentences 
def cosine_sim(token_sentence):
	average = 0
	for sentence in range(len(token_sentence)-2):
		sentence_1 = str(token_sentence[sentence])
		sentence_2 = str(token_sentence[sentence+1])
		sentence_3 = str(token_sentence[sentence+2])
		
		documents = [sentence_1, sentence_2, sentence_3]
		count_vectorizer = CountVectorizer()
		sparse_matrix = count_vectorizer.fit_transform(documents)
		doc_term_matrix = sparse_matrix.todense()
		df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names_out(), index=['sentence1', 'sentence2', 'sentence3'])
		cos_sim = cosine_similarity(df, df)
		total = 0
		for item in cos_sim:
			for value in item:
				total = total + value
			average = average + total/9	
	average_cos_sim= average/(len(token_sentence)-2)
	return average_cos_sim

# gets the word frequency of the group
def word_freq(token_sentence):
	total = FreqDist('')
	for item in token_sentence:
		fdist = FreqDist(item)
		total += fdist
	total_unique_words = len(total)
	return total_unique_words, total


# gets the pos tags for the group
def pos_tags(token_sentence):
	#nltk.download('averaged_perceptron_tagger')
	words_and_tags = []
	for item in token_sentence:
		words_and_tags.append(nltk.pos_tag(item))
	return words_and_tags

# gets the bag of words and tags for the group
def bag_of_words_and_tags(words_and_tags):
	bag_of_words = []
	bag_of_tags = []
	for item in words_and_tags:
		for pair in item:
			word, tags = pair
			bag_of_words.append(word)
			bag_of_tags.append(tags)
	return bag_of_words, bag_of_tags
	
# gets the type ratio for the group	
def type_ratio(total_unique, total):
	token_ratio = total_unique/len(total)
	return  token_ratio

# cleans the tokenized sentences for the group
def clean_token(token_sentence):
	return lemmatize_sentence(stop_words(drop_sounds(token_sentence)))
	
# gets the speakers year at YFV
def speaker_year(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['Year at UFV']

# gets the speakers gender
def speaker_gender(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['Gender']

# gets the speakers english speaking status
def speaker_english(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['English']
	
# gets the speakers individual score
def speaker_AIS(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['AIS'] 

# gets the speakers 
def speaker_AII(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['AII']
	
# gets the speakers time expectation score
def speaker_Ind_TE(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['Ind_TE']

# gets the speakers worked well together score
def speaker_Ind_WW(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['Ind_WW']

# gets the speakers time managment score
def speaker_Ind_TM(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['Ind_TM']

# gets the speakers efficiency score
def speaker_Ind_Eff(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['Ind_Eff']

# gets the speakers quality of work score
def speaker_Ind_QW(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['Ind_QW']
	
# gets the speakers satisfaction score
def speaker_Ind_Sat(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['Ind_Sat']

# gets the speakers leadership score
def speaker_Ind_Lead(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['Ind_Lead']

# gets the speakers group number
def speaker_Group_Number(corpus, speaker_id):
	return corpus.get_speaker(speaker_id).meta['Group Number']
	
# gets the speakers reply to sentence
def reply_to(corpus, utter_id):
	return corpus.get_utterance(utter_id).reply_to

# gets the time stamp of the speakers sentence
def time_stamp(corpus, utter_id):
	return corpus.get_utterance(utter_id).timestamp

# gets the speakers duration of the sentence they spoke
def duration(corpus, utter_id):
	return corpus.get_utterance(utter_id).meta['Duration']

# gets the end time of the speakers sentence
def end_time(corpus, utter_id):
	return corpus.get_utterance(utter_id).meta['End']

# gets the groups meeting size
def meeting_size(corpus, utter_id, group):
	return corpus.get_object("conversation", utter_id[group-1]).meta['Meeting Size']
	
# gets the group meeting length
def meeting_length(corpus, utter_id, group):
	return corpus.get_object("conversation", utter_id[group-1]).meta['Meeting Length in Minutes']
	
# gets the groups average group score
def AGS(corpus, utter_id, group):
	return corpus.get_object("conversation", utter_id[group-1]).meta['AGS']

# gets the group time expectation score
def group_TE(corpus, utter_id, group):
	return corpus.get_object("conversation", utter_id[group-1]).meta['Group_TE']
	
# gets the group worked well together score
def group_WW(corpus, utter_id, group):
	return corpus.get_object("conversation", utter_id[group-1]).meta['Group_WW']

# gets the group time managment score	
def group_TM(corpus, utter_id, group):
	return corpus.get_object("conversation", utter_id[group-1]).meta['Group_TM']

# gets the group efficiancy score
def group_Eff(corpus, utter_id, group):
	return corpus.get_object("conversation", utter_id[group-1]).meta['Group_Eff']
	
# gets the group quality of work score
def group_QW(corpus, utter_id, group):
	return corpus.get_object("conversation", utter_id[group-1]).meta['Group_QW']

# gets the group satisfaction score
def group_Sat(corpus, utter_id, group):
	return corpus.get_object("conversation", utter_id[group-1]).meta['Group_Sat']

# gets the please feature from the politeness features	
def get_politeness_please(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Please==']

# gets the please start feature from the politeness features	
def get_politeness_please_start(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Please_start==']

# gets the hashedge feature from the politeness features		
def get_politeness_hashedge(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==HASHEDGE==']

# gets the indirect btw feature from the politeness features		
def get_politeness_indirect(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Indirect_(btw)==']
	
# gets the hedges feature from the politeness features	
def get_politeness_hedges(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Hedges==']

# gets the factuality feature from the politeness features		
def get_politeness_factuality(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Factuality==']

# gets the deference feature from the politeness features		
def get_politeness_deference(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Deference==']

# gets the gratitude feature from the politeness features		
def get_politeness_gratitude(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Gratitude==']

# gets the apologizing feature from the politeness features	
def get_politeness_apologizing(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Apologizing==']

# gets the 1st person pl feature from the politeness features		
def get_politeness_1st_person_pl(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==1st_person_pl.==']

# gets the 1st person feature from the politeness features		
def get_politeness_1st_person(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==1st_person==']

# gets the 1st person start feature from the politeness features		
def get_politeness_1st_person_start(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==1st_person_start==']

# gets the 2nd person feature from the politeness features		
def get_politeness_2nd_person(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==2nd_person==']

# gets the 2nd person start feature from the politeness features		
def get_politeness_2nd_person_start(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==2nd_person_start==']

# gets the indirect greeting feature from the politeness features		
def get_politeness_indirect_greeting(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Indirect_(greeting)==']

# gets the direct_question feature from the politeness features		
def get_politeness_direct_question(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Direct_question==']

# gets the direct_start feature from the politeness features		
def get_politeness_direct_start(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==Direct_start==']

# gets the haspositive feature from the politeness features		
def get_politeness_has_positive(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==HASPOSITIVE==']

# gets the hasnegative feature from the politeness features		
def get_politeness_has_negative(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==HASNEGATIVE==']

# gets the subjunctive feature from the politeness features		
def get_politeness_subjunctive(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==SUBJUNCTIVE==']

# gets the indicative feature from the politeness features		
def get_politeness_indicative(corpus, utter_id):
	politness = corpus.get_utterance(utter_id).meta['politeness_strategies']
	return politness['feature_politeness_==INDICATIVE==']

# puts all politeness features together and returns average politeness score for each conversation		
def get_politeness(corpus, utter, group):
	total = 0
	total_utter = 0
	for utter_id in utter:
		if int(get_data_frame.group(utter_id)) == int(group):
			total += get_politeness_please(corpus, utter_id)
			total += get_politeness_please_start(corpus, utter_id)	
			total += get_politeness_hashedge(corpus, utter_id)	
			total += get_politeness_indirect(corpus, utter_id)
			total += get_politeness_hedges(corpus, utter_id)
			total += get_politeness_factuality(corpus, utter_id)
			total += get_politeness_deference(corpus, utter_id)
			total += get_politeness_gratitude(corpus, utter_id)
			total += get_politeness_apologizing(corpus, utter_id)
			total += get_politeness_1st_person_pl(corpus, utter_id)
			total += get_politeness_1st_person(corpus, utter_id)
			total += get_politeness_1st_person_start(corpus, utter_id)
			total += get_politeness_2nd_person(corpus, utter_id)	
			total += get_politeness_2nd_person_start(corpus, utter_id)	
			total += get_politeness_indirect_greeting(corpus, utter_id)	
			total += get_politeness_direct_question(corpus, utter_id)
			total += get_politeness_direct_start(corpus, utter_id)
			total += get_politeness_has_positive(corpus, utter_id)
			total += get_politeness_has_negative(corpus, utter_id)
			total += get_politeness_subjunctive(corpus, utter_id)	
			total += get_politeness_indicative(corpus, utter_id)
			total_utter += 1
	return total/total_utter
	
