# Transcendence Transformer

We adapt the code from [nanoGPT](https://github.com/karpathy/nanoGPT) to train a small transformer on game Set data. The aim of our experiments is to test how transcendence and in-context learning interact. Specifically, we look to see if and in what conditions a model can achieve transcendence on our game Set data by varying the model size, number in shots for in-context learning and sampling temperature. 

# Setup:
(1) Clone the repository recursively
(2) Install nanoGPT/transcendenceGPT requirements

To train the models in our paper, run the following commands: 
```
cd transcendence

# for the small model
python train.py config/small_transcendence_gpt_50_train_50_test.py

# for the big model
python train.py config/big_transcendence_gpt_50_train_50_test.py

```

To generate in distribution and out of distribution data, follow the example in ```faulty_sets/train_test_sets.py```

To evaluate the model you can use ```eval_model.py config/<your model config>``` for simple evaluation or ```eval_model_icl.py config/<your model config>``` for in context-learning evaluation. Make sure to change the dataset variable to your testing dataset folder accordingly. 

In our examples, we generate 60,000 training samples and 2,000 testing samples of length 50, in distribution and out of distribution, and evaluate the model for k-shot learning, with k from 0 to 49.

