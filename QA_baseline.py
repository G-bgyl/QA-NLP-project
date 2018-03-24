## Question Answering
# Mengying Zhang, Alicia Ge
## Baseline method

# Versions:
#### M: added read in data

# Library
import json
import pprint

# global variables listed here
QA_TYPE_MATCH = {}  # a dictionary maps question type to answer type


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

def make_ngrams(paragraph, ngram=1):
    '''
      Input: s string represents paragraph or question, ngram = 1 will make unigram, otherwise make bigram.
      Output:  list of tokens.(if not unigram, element would be tuples)
    '''


    pass


# -------------
#  Compute similarity
# -------------
def make_score(token_paragraph, token_question):
    '''
      Input: a list of list represent a paragraph;
            a list of string or tuple represent a sentence;
      Output : a sorted dictionary of score represent similarity between each sentence and question. Key: Value:score (0-1)
    '''
    pass


def answer_type(token_question):
    global QA_TYPE_MATCH

    '''
      Input: a list of String represent a question;
      Output: a string POS target type. Default NN.
    '''
    return 'NN'


def parse(sentence, atype):
    '''
      Input: sentence: one sentence
             atype: a string, target POS tag
      Function: loop through sentences from high score to low and find first word of answer type
      Output: string of answer. if fail to find, return None

    '''


def retrieve_answer(passage, questions):
    '''
      Input: string of passage and question
      Output: answer
    '''
    # Step0: prepare paragraghs to unigrams and bigrams
    para_token_set = (make_ngrams(paragraph), make_ngrams(paragraph, ngram=2))

    answer_list = []
    for question in questions:
        # Step1: questions to unigrams and bigrams
        question_token_set = (make_ngrams(question), make_ngrams(question, ngram=2))

        # Step2: window slide to find the match score between the passage sentence and the question
        score_sorted = make_score(para_token_set, question_token_set)

        # Step3:
        atype = answer_type(question_token_set[0])

        # if the top scored sentence does not contain the target answer type, go to the next sentence.
        answer = ''
        for each in score_sorted:
            answer = parse(para_token_set[each], atype)
            if answer:
                break
        if not answer:
            answer = 'Did not find answer'
            print('Did not find answer')
        answer_list.append(answer)

    return answer_list
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    train_dict = read_data("train-v1.1.json")

    # pprint.pprint(train_dict)
    #pprint.pprint(type(train_dict['data']))

    for QA_dict in train_dict['data']:
        for QA_article in QA_dict['paragraphs']:

            paragraph = QA_dict['context']
            pprint.pprint(paragraph)
            exit()


