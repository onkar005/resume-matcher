# pip install spacy
# python -m spacy download en_core_web_sm
import string
import spacy
from two_lists_similarity import Calculate_Similarity as cs
from scipy import spatial
from collections import Counter
import math
from statistics import mean
import docx2txt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from pdftotxt import pdf2txt
import textract
from manual_list_match import compare_skills_list
from spacy.matcher import PhraseMatcher


# Load English tokenizer, tagger, parser, and word vectors
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)

# Process whole documents

def spacy_match(res,jd,role):
	score_22= []
	common_score_list =[]
	not_common_list = []
	list_xlabels = []
	list_resume_values = []
	list_jd_value=[]
	
	for i in res:
		print("this is i")
		print(i)
		if i.filename.split(".")[1]=="pdf":
			text = pdf2txt(i).replace('/n',' ').replace('•',' ').replace(',',' ')
			#print(text)
		elif i.filename.split(".")[1]=="pptx" or i.filename.split(".")[1]=="ppt":
			text = textract.process(i).replace('/n',' ').replace(',',' ') 
		else:
			text = docx2txt.process(i).replace(',',' ')
		text_2= docx2txt.process(jd)
		
		doc = nlp(text)
		#print(doc)
		doc2 = nlp(text_2)
		manual_not_present_list,manual_present_list,phrase_list=compare_skills_list(role,doc)

		resume_2_2 = [chunk.text for chunk in doc.noun_chunks]
		jobDesc_2_2 = [chunk.text for chunk in doc2.noun_chunks]

#converting spacy created list to text again
		
		resume_2 = (" ".join(resume_2_2))
		jobDesc_2 = (" ".join(jobDesc_2_2))
		############addition########
		to_check = ["JJ","NNS","NN","NNP"] #"VBZ","VB","RB",
		#to_check=['ADJ']
		###########Resume conversion##########

		inter_res =[] 	# intermediate resume list

		for i in resume_2.strip().split():
			inter_res.append(i)

		inter_res_nltk = nltk.pos_tag([i for i in inter_res if i])
		#print(inter_res_nltk)


		inter_final_res=[]
		i = 0
		while i<len(inter_res_nltk):
			if inter_res_nltk[i][1] in to_check:
				inter_final_res.append(inter_res_nltk[i][0])
			else:
				pass
			i=i+1

		inter_final_res_2 = [i for i in inter_final_res if len(i) > 3]	

		resume_3 = (" ".join(inter_final_res_2))



		#######JD conversion ###########33
		inter_jd =[] 	# intermediate jd list

		for i in jobDesc_2.strip().split():
			inter_jd.append(i)

		inter_jd_nltk = nltk.pos_tag([i for i in inter_jd if i])

		inter_final_jd=[]
		i=0
		while i<len(inter_jd_nltk):
			if inter_jd_nltk[i][1] in to_check:
				inter_final_jd.append(inter_jd_nltk[i][0])
			else:
				pass
			i=i+1
		inter_final_jd_2 = [i for i in inter_final_jd if len(i) > 3]
		jobDesc_3 = (" ".join(inter_final_jd_2))

		####################


		text = [resume_3,jobDesc_3]

		#print(text)
		#######sklearn

		cv = CountVectorizer()
		#print("tbis is cv ")
		#print(cv)
		count_matrix = cv.fit_transform(text)
		#print("/n this is count matrix")
		#print(count_matrix)

		#print("/n Similarity Scores: ")
		#print(cosine_similarity(count_matrix))


		# Match percentage

		matchPercentage = cosine_similarity(count_matrix)[0][1]*100
		matchPercentage = round(matchPercentage,2)
		#print(matchPercentage)
		
		X_train_counts = cv.fit_transform(text)


		x = pd.DataFrame(X_train_counts.toarray(),columns=cv.get_feature_names(),index=['resume','JD'])
		
		y = x.transpose()
		#print(y)
		#print("this is largest/n")
		plot_data=y.nlargest(10,['JD'])
		list_jd_top = list(plot_data["JD"])
		list_resume_top=list(plot_data["resume"])
		list_index = list(plot_data.index)


		jd_1 = []
		res_1= []
		label_type = ["NORP","DATE","TIME","PERCENT","CARDINAL","QUANTITY","ORDINAL","EVENT","FACILITY","MONEY","LOC","LAW","LANGUAGE"]#["GPE" , "DATE", "PRODUCT","CARDINAL","NORP","MONEY","PERSON"]
		for entity in doc.ents:
			if entity.label_ not in label_type:
				#print(entity.label_,entity.text)
				jd_1.append(entity.text)

		jobd = [i.replace("/n","") for i in jd_1]


		#print("/nthis is new line/n")
		for entity in doc2.ents:
			if entity.label_ not in label_type:
				#print(entity.label_,entity.text)
				res_1.append(entity.text)
		resum= [i.replace("/n","") for i in res_1]
		jobd =list(dict.fromkeys(jobd))
		jobd = [str(i) for i in jobd]
		resum = list(dict.fromkeys(resum))
		resum = [str(i) for i in resum]


		def word2vec(word):
		    from collections import Counter
		    from math import sqrt

		    # count the characters in word
		    cw = Counter(word)
		    # precomputes a set of the different characters
		    sw = set(cw)
		    # precomputes the "length" of the word vector
		    lw = sqrt(sum(c*c for c in cw.values()))

		    # return a tuple
		    return cw, sw, lw

		def cosdis(v1, v2):
		    # which characters are common to the two words?
		    common = v1[1].intersection(v2[1])
		    # by definition of cosine distance we have
		    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]


		list_A = resum
		#print("this is a list ::::")
		#print(list_A)
		list_B = jobd
		score=[]
		common_word_1 = []
		threshold = 0.8     # if needed
		for key in list_A:
		    for word in list_B:
		        try:
		            result = cosdis(word2vec(word), word2vec(key))
		            score.append(result)
		            if result > threshold:
		            	common_word_1.append(word)
		            	#print("the word is : {} and key is: {} by percentge: {}".format(word, key, result))
		        except IndexError:
		            pass

		mean_score = mean(score)
		mean_score_2 = round(mean_score*100,2)
		not_common = list(set(resum) - set(common_word_1))
		#not_common=list(dict.fromkeys(resum))
		not_common.extend(manual_not_present_list)
		common_word_1=list(dict.fromkeys(common_word_1))
		common_word_1.extend(manual_present_list)
		
		#percentage calculation by taking manual list into consideration
		match_perc_adjusted = round((matchPercentage*0.9 + (len(manual_present_list)*100/len(phrase_list))*0.1),2)
		score_22.append(match_perc_adjusted)
		common_score_list.append(common_word_1)
		not_common_list.append(not_common)
		list_xlabels.append(list_index)
		list_resume_values.append(list_resume_top)
		list_jd_value.append(list_jd_top)

	return score_22,not_common_list, common_score_list,list_xlabels,list_resume_values,list_jd_value



if __name__ == "__main__":
	loc1 = [""]
	loc2=""
	#print(spacy_match(loc1,loc2))

#csObj = cs(jd, res)


#x = csObj.fuzzy_match_output()

#print(csObj.similar_input_items())






