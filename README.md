# Object-Placement
First, you must use `fix_file.py` to generate the corpus with bigrams.

To generate all of the train and test files:

`python3 object_placement_turk.py`

The resulting `csv` files in the directory will be labeled `n_train.csv` or `n_test.csv` depending on the set that it belongs to.

Here's how each set is generated:  

1:  
Train: Only 4/4 agree cases, 75% of original data randomly sampled. All four of those cases are merged into one.  
Test: Only 4/4 agree cases, the other 25% of the data. All four of those cases are merged into one.

2:  
Train: Same as before  
Test: Remove all 4/4 cases from original big dataset.

3:  
Train: Same as before  
Test: Take the cases of the test set of 1 (all 4/4) and add four of those to the test set of 2.

4:  
Train: Remove all instances that have a 'similar' item anywhere in them.  
Test: Put all of those instances that had a 'similar' item in the test set.

5:  
Train: 75% of everything, randomly sampled  
Test: 25% of everything, randomly sampled



Then, you can use
`python3 word2vec_train.py`  
to generate all of the models, which will be saved to separate directories.
`python3 word2vec_test.py` 
to load and evaluate all seven models on all five train/test sets, for both cosine and output vector metrics.
The results can be found in `accs_final.csv`. The cosine similarity accuracy comes first, then the output vector one.
One line is printed for each of the models. After each set of six models, the results for the next dataset are printed.
