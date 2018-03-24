## Question Answering
# Mengying Zhang, Alicia Ge
## Baseline method

# Versions:
#### M: added read in data


# Library
import json
import pprint


# -------------
# Read in data
# -------------
def read_data(filename):
    """ Read json formatted file, and return a dictionary """

    with open(filename) as json_data:
        doc = json.load(json_data)
        return doc

# -------------
# Compare similarity between question to each sentence
# -------------

def parse_paragraph():
    pass

def find_target_sentence():
    pass

# -------------
# Mark the type of question
# -------------

def question_type():
    pass



# ------------------------------------------------------------------------------
if __name__ == "__main__":
    train_dict = read_data("train-v1.1.json")

    # pprint.pprint(train_dict)
    pprint.pprint(type(train_dict['data']))
    i=0
    for QA_dict in train_dict['data'][0]['paragraphs']:
        i+=1
        # pprint.pprint(QA_dict)
        print('one qa dict')
        pprint.pprint(QA_dict['qas'])
        print('------------------------------',i)
        exit()