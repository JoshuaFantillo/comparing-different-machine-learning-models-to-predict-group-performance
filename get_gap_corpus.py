################################################################
# Author: Joshua Fantillo
# File: get_gap_corpus.py
# Date: April 11, 2022
# Employer: The University of the Fraser Valley
# Location: Abbotsford, BC, Canada
# Description: This file downloads the GAP-corpus from convokit
# and gets the uttereances and speaker ids for each group member
# as well as returning the whole corpus
################################################################

import convokit
from convokit import Corpus, download

# downloads the corpus and gets the utterances
def gap():
	corpus =  Corpus(download('gap-corpus'))
	utter = utt(corpus)
	print()
	return corpus, utter
	
# gets the uttereances ids
def utt(corpus):
	return corpus.get_utterance_ids()

# gets the speaker ids
def speaker_id(corpus, utter):
	speaker_id = []
	for item in utter:
		speaker = corpus.get_utterance(item).speaker.id
		if speaker not in speaker_id:
			speaker_id.append(speaker)	
	return speaker_id
