### DeBERTaQA

We presents the implementation of the **DeBERTa-v1** model from scratch and its fine-tuning for the question answering task.
- The work is based on the paper [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
- The model, as well as the basic functions required for data preprocessing, training and evaluation, are **implemented using only the PyTorch library**
- We skip the pre-training step and load the already pre-trained [DeBERTa-v1](https://huggingface.co/microsoft/deberta-base) weights from the ```transformers``` library
- We use [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) as a training dataset in fine-tuning step. The training sample size is 10 000. The test sample size is 1000
- We train only the LoRA layers with rank 32. The number of trainable parameters is 1.3% of the all parameters
- We achieved an **f1 score of about 81%** on the test after 15 training epochs.

![training history](https://github.com/user-attachments/assets/fe94fb6f-df50-4c19-8397-532eca63ea22)
