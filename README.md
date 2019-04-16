# Object-Placement

How to evaluate:

To generate all of the train and test files:

`python3 object_placement_turk.py`

Then, you can use
`python3 word2vec_test.py` 
to train and evaluate all six models on all five train/test sets, for both cosine and output vector metrics.
The results can be found in `accs_final.csv`. The cosine similarity accuracy comes first, then the output vector one.
One line is printed for each of the models. After each set of six models, the results for the next dataset are printed.
