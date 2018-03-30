# SQuAd-NLP-project
This is a project of Menagying Zhang and Alicia Ge for Umich SI 630 final project 2018.

## Baseline Overview
For now we just tried to build up a baseline.

### Methodology

We divide the problem into two. First, find the candidate sentence within a paragraph that contains the answer. Second, try to retrieve answer from the candidate sentence.

For the first step, we used both unigram and bigram method to compute the similarity between question and each sentence,and give each sentece a score. We chose two sentences of highest score to as the candidate sentence to do the later step.

For the second step, we categorize the question type and map each question type to a POS tag. We use parse to get a parsing tree for each candidate sentence and retrive all phrases that response to the question type. We then remove a phrase if it contains words show up in the question. An then randomly generated answer from these prases.

### Result

