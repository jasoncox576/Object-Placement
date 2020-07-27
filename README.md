# Object-Placement
**Generating Text Corpus with Bigram Tokens:**    
To generate a corpus with bigrams, you will need to run `fix_file.py` with the path in the source pointing to the desired text corpus.

**Generating Train/Test sets:**   
To generate all of the train and test files, run

`python3 data_gen.py`

The resulting `csv` files in the directory will be labeled `n_train.csv` or `n_test.csv` depending on the set that it belongs to.

Here's how each set is generated:  

1:  
Train: Only 4/4 agree cases, 75% of original data randomly sampled. All four of those cases are merged into one.   
Test: Only 4/4 agree cases, the other 25% of the data. All four of those cases are merged into one.

2:  
Train: Same as previous `1_train.csv`.       
Test: All of the 3/4 agreement cases from the original dataset. Expect slightly lower accuracy when testing on this set. 

3:  
Train: Same as before, still 75% of 4/4 cases.    
Test: All of the 2/4 agreement cases. Because there's so much disagreement among annotators, expect significantly hindered accuracy of no fault of the model.

4:  
Train: Remove all instances that have a 'similar' item anywhere in them.  
Test: Put all of those instances that had a 'similar' item in the test set.

5:
Train: 75% of everything, randomly sampled  
Test: 25% of everything, randomly sampled



**Training and Evaluation**

Then, you can use
`python3 word2vec_train.py`  
to generate all of the models, trained with validation, which will be saved to separate directories.
`python3 word2vec_test.py` 
to load and evaluate all models on all train/test sets.
The results can be found in `accs_final.csv`.
