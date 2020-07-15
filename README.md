### file details
####  the dir of conceptnet is about code of concepnet operation and the conceptnet data I have used.
- common.py: contain some variables and function which would be used frequently in other files
- rearrange_conceptnet: loading conceptnet into memory
- get_concept: get target triple of each choice
#### the dir of model is about the code of my model, the most useful file is model.py and layers.py
#### the dir of utils is about some core file about my model,
#### the dir of specific about some code of data preparations
#### the dir of bash contain many sh files , and task.sh is the main file, when sh task.sh, the training will be started.
#### task.py is the main file of my model and the training file
#### task_predict is the test file that given a trained model and the test file, it can output the prediction initial csv
#### get_test_results is the file that given the prediction initial csv , it can output the prediction csv which contains ids and answers, 
