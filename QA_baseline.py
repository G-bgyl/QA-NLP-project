## Question Answering
# Mengying Zhang, Alicia Ge
## Baseline method

# Versions:
#### M: added read in data

# Library
import json
import pprint
import nltk
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import numpy as np
import operator
import csv



# global variables listed here
QA_TYPE_MATCH = {'what':'NN','when':'NN','how':'NN','where':'NN','whom':'NN','why':'NN','who':'NN','which':'NN','whose':'NN','name':'NN','example':'NN'}  # a dictionary maps question type to answer type
tokenizer = RegexpTokenizer(r'\w+')

# -------------
# Read in data
# -------------
def read_data(filename):
    """ Read json formatted file, and return a dictionary """

    with open(filename) as json_data:
        doc = json.load(json_data)
        return doc


# -------------
#  Contextualize paragrapgh
# -------------

def make_ngrams(paragraph, ngrams=[1]):
    '''
      Input: s string represents paragraph or question, ngram = 1 will make unigram, otherwise make bigram.
      Output:  list of tokens.(if not unigram, element would be tuples)
    '''

    sent_tokenize_list = sent_tokenize(paragraph)
    token_p={}


    for ngram in ngrams:
        if ngram ==1:
            token_p['1']=[]
            for sent in sent_tokenize_list:

                token_sent =tokenizer.tokenize(sent)

                token_p['1'].append(token_sent)
        elif ngram==2:
            token_p['2'] = []
            for sent in sent_tokenize_list:
                token_sent = tokenizer.tokenize(sent)
                bi_token_sent =list(nltk.bigrams(token_sent))

                token_p['2'].append(bi_token_sent)

        else:
            token_p[str(ngram)]=[]
            # haven't test
            for sent in sent_tokenize_list:
                token_s = tokenizer.tokenize(sent)


                new_ngram = []
                i = 0
                for i in range(len(token_s)-ngram):
                    new_ngram.append(list(token_s[j] for j in range(i,i+ngram)))
                token_p[str(ngram)].append(new_ngram)

    return token_p



# -------------
#  Compute similarity
#  main function: make_score
#  subfunction: uni_score, bi_score
# -------------

#subfunction of make_score
def uni_score(token_paragraph_uni,token_question_uni):
    uni_raw_score_list = []
    # loop through each sent
    for sent_p in token_paragraph_uni:
        raw_score_uni = 0
        # loop through each unigram
        for word_s in sent_p:
            for word_q in token_question_uni:
                if word_s == word_q:
                    raw_score_uni += 1
        uni_raw_score_list.append(raw_score_uni)
    return uni_raw_score_list

#subfunction of make_score
def bi_score(token_paragraph_bi,token_question_bi):
    bi_raw_score_list = []
    # loop through each sent
    for sent_p in token_paragraph_bi:
        raw_score_bi = 0
        # loop through each unigram
        for word_s in sent_p:
            for word_q in token_question_bi:
                if word_s == word_q:
                    raw_score_bi += 1
        bi_raw_score_list.append(raw_score_bi)
    return bi_raw_score_list

def make_score(token_paragraph, token_question):
    '''
      Input: a list of list represent a paragraph;
            a list of string or tuple represent a sentence;
      Output : a sorted dictionary of score represent similarity between each sentence and question. Key: Value:score (0-1)
    '''
    len_q = len(token_question['1'][0])
    uni_raw_score_list = np.array(uni_score(token_paragraph['1'],token_question['1'][0]))
    bi_raw_score_list = np.array(bi_score(token_paragraph['2'], token_question['2'][0]))
    score_list = (1/3 *uni_raw_score_list + 2/3 * bi_raw_score_list)/len_q
    score_dict={}
    for i in range(len(score_list)):
        score_dict[i] = score_list[i]
    score_dict=sorted(score_dict.items(), key=operator.itemgetter(1),reverse=True)

    return score_dict

def answer_type(token_question):
    global QA_TYPE_MATCH
    '''
      Input: a list of String represent a question;
      Output: a string POS target type. Default NN.
    '''
    aType = None
    contain = False

    for qMark in QA_TYPE_MATCH:
        if qMark in token_question:
            aType = qMark
            contain = True
            break
    if not contain:
        pass

    return aType


def parse(sentence, atype):

    '''
      Input: sentence: one sentence
             atype: a string, target POS tag
      Function: loop through sentences from high score to low and find first word of answer type
      Output: string of answer. if fail to find, return None

    '''


def retrieve_answer(paragraph, questions):
    '''
      Input: string of passage and question
      Output: answer
    '''

    # Step0: prepare paragraghs to unigrams and bigrams
    para_token_set = (make_ngrams(paragraph,ngrams=[1,2]))

    answer_list = []
    untrack = 0
    for question in questions:

        # Step1: questions to unigrams and bigrams
        question_token_set = (make_ngrams(question,ngrams=[1,2]))

        # Step2: window slide to find the match score between the passage sentence and the question
        score_sorted = make_score(para_token_set, question_token_set)
        # print the highest sentence score
        # pprint.pprint(score_sorted[0][1])


        # Step3:
        atype = answer_type(question_token_set['1'])
        if atype is None:
            untrack+=1
        '''
        # if the top scored sentence does not contain the target answer type, go to the next sentence.
        answer = ''
        for each in score_sorted:
            answer = parse(para_token_set[each], atype)
            if answer:
                break
        if not answer:
            answer = 'Did not find answer'
            print('Did not find answer')
        answer_list.append(answer)'''


    return answer_list
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    train_dict = read_data("train-v1.1.json")

    #for intuition:
    test_question = []
    untrack =0
    #loop through all articles
    for QA_dict in train_dict['data']:
        # loop through all paragraphs
        for QA_article in QA_dict['paragraphs']:
            paragraph = QA_article['context'].lower()
            questions = []
            # loop through all qas
            for qa in QA_article['qas']:
                questions.append(qa['question'].lower())

                # for intuition:
                test_question.append(qa['question'].lower())

        untrack += retrieve_answer(paragraph, questions)
    print(untrack)
    # output file to get intuition of questions.
    '''with open('all_question.csv', 'w') as all_question, open('untrack_question.csv', 'w') as untrack_question:
        writer = csv.writer(all_question)
        writer2 = csv.writer(untrack_question)

        untrack = 0
        for question in test_question:
            contain =False
            writer.writerow(question)
            for qMark  in QA_TYPE_MATCH:
                if qMark in question:
                    contain = True
                    break
            if not contain:
                writer2.writerow(question)'''






