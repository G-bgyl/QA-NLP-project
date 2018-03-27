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
from nltk.parse.stanford import StanfordParser
from random import *

# global variables listed here
QA_TYPE_MATCH = {'what': 'NP', 'when': 'CD', 'where': 'NP', 'whom': 'NP', 'why': 'NP',
                 'who': 'NP', 'which': 'NP', 'whose': 'NP', 'name': 'NP', 'example': 'NP', 'how many': 'CD',
                 'how often': 'CD','what year':'CD','location':'NP'}  # a dictionary maps question type to answer type
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
    token_p = {}

    for ngram in ngrams:
        if ngram == 1:
            token_p['1'] = []
            for sent in sent_tokenize_list:
                token_sent = tokenizer.tokenize(sent.lower())

                token_p['1'].append(token_sent)
        elif ngram == 2:
            token_p['2'] = []
            for sent in sent_tokenize_list:
                token_sent = tokenizer.tokenize(sent.lower())
                bi_token_sent = list(nltk.bigrams(token_sent))

                token_p['2'].append(bi_token_sent)

        else:
            token_p[str(ngram)] = []
            # haven't test
            for sent in sent_tokenize_list:
                token_s = tokenizer.tokenize(sent.lower())

                new_ngram = []
                i = 0
                for i in range(len(token_s) - ngram):
                    new_ngram.append(list(token_s[j] for j in range(i, i + ngram)))
                token_p[str(ngram)].append(new_ngram)

    return sent_tokenize_list, token_p


# -------------
#  Compute similarity
#  main function: make_score
#  subfunction: uni_score, bi_score
# -------------

# subfunction of make_score
def uni_score(token_paragraph_uni, token_question_uni):
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


# subfunction of make_score
def bi_score(token_paragraph_bi, token_question_bi):
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
    uni_raw_score_list = np.array(uni_score(token_paragraph['1'], token_question['1'][0]))
    bi_raw_score_list = np.array(bi_score(token_paragraph['2'], token_question['2'][0]))
    score_list = (1 / 3 * uni_raw_score_list + 2 / 3 * bi_raw_score_list) / len_q
    score_dict = {}
    for i in range(len(score_list)):
        score_dict[i] = score_list[i]
    score_dict = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)

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

        if qMark in token_question[0]:
            aType = QA_TYPE_MATCH[qMark]
            contain = True
            break
    if not contain:
        pass

    return aType


# subfunction of parse
def ExtractPhrases(myTree, phrase):
    # Tree manipulation from https://www.winwaed.com/blog/2012/01/20/extracting-noun-phrases-from-parsed-trees/
    # Extract phrases from a parsed (chunked) tree
    # Phrase = tag for the string phrase (sub-tree) to extract
    # Returns: List of deep copies;  Recursive
    myPhrases = []
    if (myTree.label() == phrase):
        myPhrases.append(myTree.copy(True))
    for child in myTree:
        if (type(child) is nltk.tree.Tree):
            list_of_phrases = ExtractPhrases(child, phrase)
            if (len(list_of_phrases) > 0):
                myPhrases.extend(list_of_phrases)
    return myPhrases


def parse(sentence, atype, parser):
    '''
      Input: sentence: two sentences tuple
             atype: a string, target POS tag
             parser: use Stanford coreNLP parser, defined in main
      Function: loop through sentences from high score to low and find first word of answer type
      Output: string of answer. if fail to find, return None
    '''
    s1, s2 = sentence
    # s1 = 'it is a replica of the grotto at lourdes, france where the virgin mary reputedly appeared to saint bernadette soubirous in 1858.'
    # print ("Sent: ",sentence)
    # print ("s1: ",s1)
    # print ("s2: ", s2)
    potential_answer = []
    # atype = "NP"

    # parse s1
    assert (s1)
    result = list(parser.raw_parse(s1))
    tree = result[0]

    list_of_phrases = ExtractPhrases(tree, atype)
    potential_answer = []
    for phrase in list_of_phrases:
        # print (">> ", phrase.leaves())
        potential_answer.append(" ".join(phrase.leaves()))
        # print ("PA:>> "," ".join(phrase.leaves()))

    # if atype not found in s1, parse s2
    if len(potential_answer) == 0:
        if s2 is None:
            return None
        else:
            # print ("@S2")
            result = list(parser.raw_parse(s2))
            tree = result[0]
            list_of_phrases = ExtractPhrases(tree, atype)
            for phrase in list_of_phrases:
                # print (">> ", phrase.leaves())
                potential_answer.append(" ".join(phrase.leaves()))

        if len(potential_answer) == 0: return None

    potential_answer.sort(key=len)
    rm = []  # long sentences to remove

    for i in range(len(potential_answer)):
        if i == len(potential_answer): break
        for j in range(i + 1, len(potential_answer)):
            if potential_answer[i] in potential_answer[j]:
                rm.append(potential_answer[j])
                # print ("RM: ", potential_answer[j])
    nrrw = [fruit for fruit in potential_answer if fruit not in rm]  # narrowed candidates

    # for i in nrrw:
    # print ("nrrw: >> ",i)

    # *randomly return one answer
    bef_length = len(potential_answer)
    aft_length = len(nrrw)
    rmd_index = randint(1, aft_length) - 1
    answer = nrrw[rmd_index]
    print("Before removing, we have: ", bef_length, "After we have: ", aft_length, '\n')
    # print ("Answer: ", answer)
    # exit()
    return (answer)


def retrieve_answer(paragraph, questions, parser):
    '''
      Input: string of passage and question
      Output: answer
    '''

    # Step0: prepare paragraghs to unigrams and bigrams
    sent_tokenize_list, para_token_set = (make_ngrams(paragraph, ngrams=[1, 2]))

    answer_list = []
    sent_list = []
    untrack = 0
    for question in questions:

        # Step1: questions to unigrams and bigrams
        unused_question_list, question_token_set = (make_ngrams(question, ngrams=[1, 2]))

        # Step2: window slide to find the match score between the passage sentence and the question
        score_sorted = make_score(para_token_set, question_token_set)
        print('question:',question)
        print('sentence:',sent_tokenize_list[score_sorted[0][0]])
        if len(score_sorted) > 1:
            candidate_sent = (sent_tokenize_list[score_sorted[0][0]], sent_tokenize_list[score_sorted[1][0]])
        else:
            candidate_sent = (sent_tokenize_list[score_sorted[0][0]], None)
        # for parse:

        # candidate_sent = (sent_tokenize_list[score_sorted[0][0]],sent_tokenize_list[score_sorted[1][0]])

        # for test sentence retrival accuracy
        if len(score_sorted) > 1:
            sent_list.append((sent_tokenize_list[score_sorted[0][0]], sent_tokenize_list[score_sorted[1][0]]))
        else:
            sent_list.append((sent_tokenize_list[score_sorted[0][0]], ''))

        # Step3:
        atype = answer_type(question_token_set['1'])

        if atype is None:
            untrack += 1

        # if the top scored sentence does not contain the target answer type, go to the next sentence.

        answer = parse(candidate_sent, atype, parser)
        print('answer:', answer)
        answer_list.append(answer)

    return answer_list


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # for parse:

    path_to_models_jar = '/Users/G_bgyl/si630/project/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar'  # change to your path

    path_to_jar = '/Users/G_bgyl/si630/project/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar'  # change to your path

    parser = StanfordParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

    train_dict = read_data("dev-v1.1.json")

    # for intuition:
    # test_question = []
    untrack = 0
    right = 0
    wrong = 0
    # loop through all articles
    for QA_dict in train_dict['data']:
        # loop through all paragraphs
        for QA_article in QA_dict['paragraphs']:
            paragraph = QA_article['context']
            questions = []
            answers = []
            # loop through all qas
            for qa in QA_article['qas']:
                questions.append(qa['question'])
                for answer in qa['answers']:
                    answers.append(answer['text'])

                # for intuition:
                # test_question.append(qa['question'])

            # for test sentence retrival accuracy
            answer_list = retrieve_answer(paragraph, questions, parser)
            for i in range(len(answer_list)):

                find = False

                if answers[i]:
                    if answer_list[i]:
                        if answers[i] in answer_list[i] or answer_list[i] in answers[i] or answer_list[i] == answers[i]:
                            right += 1
                            print('Yay!')
                            print('right answer:', answers[i], 'our answer:', answer_list[i])
                            print('question:', questions[i], '\n')
                            find = True
                            continue
                    else:
                        print('answer_list[',i,'] went wrong')
                        print(len(answer_list),answer_list)
                else:
                    print('answers[', i, '] went wrong')


                if not find:
                    wrong += 1
                    print('right answer:', answers[i])
                    print('our answer:', answer_list[i])
                    print('question:',questions[i],'\n')

    print(right)
    print(wrong)
    print('sentence retrival accuracy:', right / (right + wrong))



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
