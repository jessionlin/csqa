# Albert + KCR(knowledge chosen by relations)

### Input schema

- We find the path from question_concept to choice_concept in conceptnet, if we find it , the input schema will be [CLS] stem [SEP]  question_concept  relation  choice_concept  [SEP] , if the path is not unique, then we choose the first one with the shortest path . For example: [CLS] A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? [SEP] revolving door AtLocation bank [SEP]

- If we can't find the path from question_concept to choice_concept, but the choice_concept can find other triples in the conceptnet , then we use a score function to compute the final score of each triples , and we chose the triple with highest score, the followings are how we define the score function:

  - each triple in the conceptnet has a weight , and we define the weight as $tripleweight$

  - each concept will get several triples from conceptnet , and this triples can be grouped in several relations ,and the number of  the type of this  relations will no more than 34. Then we count the number of each group of relations. the formulate are following: 

     $weight_{rel_i} = \frac{1.0}{\frac{N_i}{N}} = \frac{N}{N_i}$

     $score_{triple_j} = tripleweight_{triple_j} * weight_{rel_i}$

    where $i$ is remarked as the  $i$th relation and $j$ is remarked as the $j$th triple, and $N_i$ remarked the number of triples whose relation is $rel_i$, and $N$  remark the number of all the triples gained from conceptnet. So ,  $weight_{rel_i}$ represent the weight of $rel_i$, while $tripleweight_{triple_j}$represent the weight of $triple_j$. 

  Finally we choose the triple with highest $score_{triple_j}$. And the input schema will be  [CLS] stem [SEP]  choice_concept  relation object  [SEP]. For example : [CLS] A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? [SEP] library PartOf  house [SEP]
  
- If we either find the the path from question_concept to choice_concept nor the triples of choice_concept in conceptnet, we define the input schema as: [CLS] stem [SEP] question_concept [SEP] choice_concept [SEP]. 

### Model

We propose a knowledge base method to enhance text encoder. The text encoder of our model is Albert. We feed the input into text encoder , and the output  is fed into a linear classifier to get the prediction.

### Hyper-parameters Setting

Learning Rate: 1e-5

Batch Size: 8

Weight Decay: 0.1

Max Sequence Length: 64

GPU: P100 16G  

Driver Version: 418.67       

CUDA Version: 10.1 
