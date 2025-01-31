## Question Answering
# Mengying Zhang, Alicia Ge
## Baseline method



# Library
import json
import pprint
import nltk
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import operator
import csv
import copy
from nltk.parse.stanford import StanfordParser
import random

# global variables listed here
QA_TYPE_MATCH = {'what': 'NP', 'when': 'CD', 'where': 'NP', 'whom': 'NP', 'why': 'NP',
                 'who': 'NP', 'which': 'NP', 'whose': 'NP', 'name': 'NP', 'example': 'NP', 'how many': 'CD','how much': 'CD',
                 'what percentage': 'CD','how often': 'CD','what year':'CD','location':'NP'}  # a dictionary maps question type to answer type
tokenizer = RegexpTokenizer(r'\w+')
random.seed(2018)



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
    aType = 'NP'
    contain = False
    for qMark in QA_TYPE_MATCH:

        if qMark in token_question[0]:
            aType = QA_TYPE_MATCH[qMark]
            contain = True
            break
    if not contain:
        pass

    return aType



def ExtractPhrases(myTree, phrase, bot = True):
    '''
       Input:
         Tree: a parsed tree
         Phrase: tag for the string phrase (sub-tree) to extract, eg. Np
         bot: weather to extract the bottom level of the tree, if false, it only retrieves from top level
       Output: List of deep copies;  Recursive
    ## Adapted from https://www.winwaed.com/blog/2012/01/20/extracting-noun-phrases-from-parsed-trees/
    '''

    myPhrases = []

    if (myTree.label() == phrase):
        myPhrases.append(myTree.copy(True))
        if bot == False: return myPhrases
    for child in myTree:
        if (type(child) is nltk.tree.Tree):
            list_of_phrases = ExtractPhrases(child, phrase, bot)
            if (len(list_of_phrases) > 0):
                myPhrases.extend(list_of_phrases)
    return myPhrases




def prepare_candidates(question, sentence, atype, parser):
    '''
      Input: question, target sentence both are string. atype is answer type. parser is stanford parser.
      Output: list of final candidates, if not found, return None
    '''

    question_token_list = nltk.word_tokenize(question)
    stop_words = list(stopwords.words('english'))
    stop_words.extend(['.', ','])

    filtered_question = [w for w in question_token_list if not w in stop_words]
    #print ("Q_token: ", filtered_question)

    #sentence = "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858."
    if len(sentence) == 0: return None

    result = list(parser.raw_parse(sentence))

    tree = result[0]

  ##### for top level
    top_level = ExtractPhrases(tree, atype, bot = False)
    top_candidates = []
    for phrase in top_level:
        top_candidates.append(" ".join(phrase.leaves()))
    top_nrrw = [fruit for fruit in top_candidates if fruit not in ['it', "It", "there", "this", "This", "There"]]
    #print (">> Top: ",top_nrrw)






  ##### Find Answer

    ## find from top level first
    final_candidates = copy.copy(top_nrrw)
    #print ("Question filtered: ", filtered_question)
    # remove overlapping ones
    for cand in top_nrrw:
        for token in filtered_question:
            if token in cand:
                final_candidates.remove(cand)
                break

    #print ("-------")
    #print (sentence)

    ## find from bottom level if top level is too broad
    if len(final_candidates) == 0:
        
        ##### for bottom level
        bottom_level = ExtractPhrases(tree, atype)
        bottom_candidates = []
        for phrase in bottom_level:
            bottom_candidates.append(" ".join(phrase.leaves()))
            # print ("PA:>> "," ".join(phrase.leaves()))

        bottom_candidates.sort(key=len)
        rm = ['it', "It", "there", "this", "This", "There"]  # long sentences to remove, we also remove 'it'

        for i in range(len(bottom_candidates)):
            if i == len(bottom_candidates): break
            for j in range(i + 1, len(bottom_candidates)):
                if bottom_candidates[i] in bottom_candidates[j]:
                    rm.append(bottom_candidates[j])

        bottom_nrrw = [fruit for fruit in bottom_candidates if fruit not in rm]  # narrowed candidates
        # print (">> Bottom: ", bottom_nrrw)

        # remove overlapping ones
        final_candidates = copy.copy(bottom_nrrw)
        for cand in bottom_nrrw:
            for token in filtered_question:
                if token in cand:
                    final_candidates.remove(cand)
                    break
        #print (">>BOTTOMfinal_candidates: ", final_candidates, "\n")


    #else: # for debug
        #print (">>TOPfinal_candidates: ", final_candidates,"\n")

    ### if didn't find anything, return none
    if len(final_candidates) == 0: return None
    else: return final_candidates


def prepare_answer(final_candidates):
    '''
      Input: list of final candidates returned from prepare_candidates()
      Output: (answer, number of candidate answers)
    '''

    # randomly return one answer
    if not final_candidates : return None, 0
    final_length = len(final_candidates)
    rmd_index = random.randint(1, final_length) - 1
    answer = final_candidates[rmd_index]
    # print("Final candidates: ", final_length)
    # print ("Answer: ", answer)
    # exit()

    return (answer, final_length)


def retrieve_answer(paragraph, questions, parser):
    '''
      Input: string of passage and question
      Output: answer
    '''

    # Step0: prepare paragraghs to unigrams and bigrams
    sent_tokenize_list, para_token_set = (make_ngrams(paragraph, ngrams=[1, 2]))

    answer_list = []
    aft_length_list=[]
    sent_list = []
    untrack = 0

    for question in questions:

        # Step1: questions to unigrams and bigrams
        unused_question_list, question_token_set = (make_ngrams(question, ngrams=[1, 2]))

        # Step2: window slide to find the match score between the passage sentence and the question
        score_sorted = make_score(para_token_set, question_token_set)
        #print('question:',question)


        # Step3: retrieve the answer using s1 or s2
        atype = answer_type(question_token_set['1'])

        if atype is None:
            untrack += 1

        # if the top scored sentence did not find the answer, go to the next sentence.
        s1 = sent_tokenize_list[score_sorted[0][0]]
        s1_candidates = prepare_candidates(question, s1, atype, parser)
        answer, aft_length = prepare_answer(s1_candidates)

        if answer == None and len(score_sorted) > 1: # if we have the 2nd sentence and s1 did not find answer
            s2 = sent_tokenize_list[score_sorted[1][0]]
            # print('>>>> sentence2')
            s2_candidates = prepare_candidates(question, s2, atype, parser)
            answer, aft_length = prepare_answer(s2_candidates)


        #print('answer:', answer, '\n')
        answer_list.append(answer)
        aft_length_list.append(aft_length)

    return answer_list, aft_length_list

def write_line(question,answer,our_answer,aft_length,correct):
    print('right answer:', answer)
    print('our answer:', our_answer)
    print('question:', question)
    print('candidate number:', aft_length, '\n')

    row = [question,answer,our_answer,aft_length,correct]
    writer.writerow(row)


def write_line(question,answer,our_answer,aft_length,correct):
    print('right answer:', answer)
    print('our answer:', our_answer)
    print('question:', question)
    print('candidate number:', aft_length, '\n')

    row = [question,answer,our_answer,aft_length,correct]
    writer.writerow(row)


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # for parse:

### Alicia
    path_to_models_jar = '/Users/G_bgyl/si630/project/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar'  # change to your path
    path_to_jar = '/Users/G_bgyl/si630/project/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar'  # change to your path

### Mengying
    # path_to_models_jar = "/Users/Mengying/Desktop/SI630 NLP/FinalProject/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar" # change to your path
    # path_to_jar = "/Users/Mengying/Desktop/SI630 NLP/FinalProject/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar" # change to your path


### CAEN
    #path_to_models_jar = 'stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar'  # change to your path
    #path_to_jar = "stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar"

    parser = StanfordParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

    train_dict = read_data("/Users/G_bgyl/si630/project/dev-v1.1.json")


    # for test output:
# for test output:
    test_output = []

    right1,right2,right3 = 0,0,0
    sum_2,right_2,sum_3,right_3=0,0,0,0
    quality_2_list,quality_3_list = [],[]
    wrong = 0
    with open('baseline_dev_result.csv', 'w') as baseline_dev_result: # , open('untrack_question.csv', 'w') as untrack_question
        writer = csv.writer(baseline_dev_result)
        writer.writerow(['Question', 'Correct Answer', 'Our Answer', 'Candidate Number','Correct'])
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
                    answers.append(qa['answers'][0]['text'])

                    # for intuition:
                    # test_question.append(qa['question'])

                # for test sentence retrival accuracy
                answer_list,aft_length_list = retrieve_answer(paragraph, questions, parser)
                for i in range(len(answer_list)):

                    find = False

                    if answers[i]:
                        if answer_list[i]:
                            # exactly correct
                            if answer_list[i] == answers[i]:
                                right1 += 1

                                print('Yay!')
                                find = True
                                write_line(questions[i], answers[i], answer_list[i], aft_length_list[i], 1)
                                continue
                            # our answer contains correct answer
                            elif answers[i] in answer_list[i]:
                                right2 += 1
                                right_2 = len(answers[i].split())
                                sum_2 = len(answer_list[i].split())
                                quality_2 = right_2/ sum_2
                                quality_2_list.append(quality_2)
                                print('Yay!')
                                find = True
                                write_line(questions[i], answers[i], answer_list[i], aft_length_list[i], 2)
                                continue
                            # correct answer contains our answer
                            elif answer_list[i] in answers[i]:
                                right3 += 1
                                right_3 = len(answer_list[i].split())
                                sum_3 = len(answers[i].split())
                                quality_3 = right_3 / sum_3
                                quality_3_list.append(quality_3)
                                print('Yay!')
                                find = True
                                write_line(questions[i], answers[i], answer_list[i], aft_length_list[i], 3)
                                continue
                        else:
                            print('answer_list[',i,'] went wrong')
                            print(len(answer_list),answer_list)
                    else:
                        print('answers[', i, '] went wrong')


                    if not find:
                        wrong += 1
                        write_line(questions[i], answers[i], answer_list[i], aft_length_list[i], 0)
                print('For one paragraph:')
                print('count of exact right:', right1)
                print('count of right with type 1 error:', right1 + right2)
                print('count of right with type 2 error:', right1 + right3)
                print('count of rough right:', right1 + right2 + right3)
                print('count of wrong:', wrong)

                print('proportion accuracy for exact right:',
                      round(right1 / (right1 + right2 + right3 + wrong), 3))
                print('proportion accuracy for right with type 1 error:',
                      round((right1 + right2) / (right1 + right2 + right3 + wrong),3))
                print('proportion accuracy for right with type 2 error:',
                      round((right1 + right3) / (right1 + right2 + right3 + wrong),3))
                print('proportion accuracy for rough right:',
                      round((right1 + right2 + right3) / (right1 + right2 + right3 + wrong),3))

                if sum_2 !=0:
                    print('quality of type 1 error:', round(sum(quality_2_list) / len(quality_2_list),3))
                    print('answer accuracy for right with type 1 error:',
                          round((sum(quality_2_list) + right1) / (len(quality_2_list) + right1) * (right1 + right2) / (right1 + right2 + right3 + wrong),3))
                else:
                    print('sum_2=0')
                if sum_3 != 0:
                    print('quality of type 2 error:', round(sum(quality_3_list) / len(quality_3_list),3))
                    print('answer accuracy for right with type 2 error:',
                          round( (sum(quality_3_list)+ right1)/(len(quality_3_list) +right1)* (right1 + right3) / (right1 + right2 + right3 + wrong),3))
                else:
                    print('sum_3=0 \n')
                if sum_2 != 0 and sum_3 != 0:

                    print('overall quality:',round((1 + right_2 / sum_2 + right_3 / sum_3)/3 ,3))
                    print('answer retrival accuracy for rough right:',
                          round( ((1 + right_2 / sum_2 + right_3 / sum_3) / 3) * (right1 + right2 + right3) / (
                                  right1 + right2 + right3 + wrong),3))
            print('For one article:')
            print('count of exact right:', right1)
            print('count of right with type 1 error:', right1 + right2)
            print('count of right with type 2 error:', right1 + right3)
            print('count of rough right:', right1 + right2 + right3)
            print('count of wrong:', wrong)

            print('proportion accuracy for exact right:',
                  round(right1 / (right1 + right2 + right3 + wrong), 3))
            print('proportion accuracy for right with type 1 error:',
                  round((right1 + right2) / (right1 + right2 + right3 + wrong), 3))
            print('proportion accuracy for right with type 2 error:',
                  round((right1 + right3) / (right1 + right2 + right3 + wrong), 3))
            print('proportion accuracy for rough right:',
                  round((right1 + right2 + right3) / (right1 + right2 + right3 + wrong), 3))

            if sum_2 != 0:
                print('quality of type 1 error:', round(sum(quality_2_list) / len(quality_2_list), 3))
                print('answer accuracy for right with type 1 error:',
                      round((sum(quality_2_list) + right1) / (len(quality_2_list) + right1) * (right1 + right2) / (
                              right1 + right2 + right3 + wrong), 3))
            else:
                print('sum_2=0')
            if sum_3 != 0:
                print('quality of type 2 error:', round(sum(quality_3_list) / len(quality_3_list), 3))
                print('answer accuracy for right with type 2 error:',
                      round((sum(quality_3_list) + right1) / (len(quality_3_list) + right1) * (right1 + right3) / (
                              right1 + right2 + right3 + wrong), 3))
            else:
                print('sum_3=0 \n')
            if sum_2 != 0 and sum_3 != 0:
                print('overall quality:', round((1 + right_2 / sum_2 + right_3 / sum_3) / 3, 3))
                print('answer retrival accuracy for rough right:',
                      round(((1 + right_2 / sum_2 + right_3 / sum_3) / 3) * (right1 + right2 + right3) / (
                              right1 + right2 + right3 + wrong), 3))
        print('For whole document:')
        print('count of exact right:', right1)
        print('count of right with type 1 error:', right1 + right2)
        print('count of right with type 2 error:', right1 + right3)
        print('count of rough right:', right1 + right2 + right3)
        print('count of wrong:', wrong)

        print('proportion accuracy for exact right:',
              round(right1 / (right1 + right2 + right3 + wrong), 3))
        print('proportion accuracy for right with type 1 error:',
              round((right1 + right2) / (right1 + right2 + right3 + wrong), 3))
        print('proportion accuracy for right with type 2 error:',
              round((right1 + right3) / (right1 + right2 + right3 + wrong), 3))
        print('proportion accuracy for rough right:',
              round((right1 + right2 + right3) / (right1 + right2 + right3 + wrong), 3))

        if sum_2 != 0:
            print('quality of type 1 error:', round(sum(quality_2_list) / len(quality_2_list), 3))
            print('answer accuracy for right with type 1 error:',
                  round((sum(quality_2_list) + right1) / (len(quality_2_list) + right1) * (right1 + right2) / (
                          right1 + right2 + right3 + wrong), 3))
        else:
            print('sum_2=0')
        if sum_3 != 0:
            print('quality of type 2 error:', round(sum(quality_3_list) / len(quality_3_list), 3))
            print('answer accuracy for right with type 2 error:',
                  round((sum(quality_3_list) + right1) / (len(quality_3_list) + right1) * (right1 + right3) / (
                          right1 + right2 + right3 + wrong), 3))
        else:
            print('sum_3=0 \n')
        if sum_2 != 0 and sum_3 != 0:
            print('overall quality:', round((1 + right_2 / sum_2 + right_3 / sum_3) / 3, 3))
            print('answer retrival accuracy for rough right:',
                  round(((1 + right_2 / sum_2 + right_3 / sum_3) / 3) * (right1 + right2 + right3) / (
                          right1 + right2 + right3 + wrong), 3))
