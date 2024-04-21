# Transformer-based Calculator

Simple exploration of basic transformer architectures. 

## Tutorial

Step-by-step jupyter notebook tutorial taken from Karpathy's Youtube series: https://www.youtube.com/watch?v=kCc8FmEb1nY

## Calculator

gen_calculator_dataset.py: generates a simple .txt file dataset containing arithmetic problems involving (+, -, *, //) and a specified min and max for integers used.

transformer.py: gpt-2 like architecture, refactored out of jupyter notbook tutorial into standalone file.

train.py: trainer for a given dataset, output looks like:
'''
Training config:  Config(batch_size=64, num_iterations=5000, lr=0.0003, eval_interval=100, block_size=12, vocab_size=17, n_layer=6, n_head=6, n_embed=384, dropout=0.1)
Dataset location:  math.txt
Training set size:  1209022
Validation set size:  134336

Fresh model sample:
 +92010518/57-8+35++
65
4*=4/+=** 3/6+ 7++87-41-6+=5

Training model...
Estimated losses: 1.27914 train, 1.28612 val: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [31:35<00:00,  2.64it/s]

Trained sample:
  + 3 = 52
57 // 74 = 0
68 * 26 = 1126
22 * 24 = 54
48 // 28 = 1
39 // 82 = 0
12 - 26 = -2
48 // 59 = 1
'''
