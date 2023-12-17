# Final project for "Numerical Linar Algebra" master's course

### Abstract

LoRA - Low Rank Adaptation of LLM - is one of the most popular methods for efficient fine-tuning of Large Language Models without need for training all of it's parametrs. One of it's most recent modifications - LoRA-FA is a memory efficient extension of the method, which allows to reduce memory cost around x1.4 times. However both LoRA and LoRA-FA rely on random initialization of matrix A and initialization with zeros of matrix B. In our project we plan to validate LoRA-FA paper results and extend them via experiments with intialization of learnable low-rank matrix decomposition (for example, via computing SVD of original weight matrix) and additional loss regularization on distance from original weights decomposition in order to speed up the fine-tuning process.

### Datasets from article
"CoLA":'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',

"SST":'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',

"QQP":'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',

"STS":'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',

"MNLI":'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',

"QNLI":'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',

"RTE":'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',

"WNLI":'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',

"diagnostic":'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'}

### How to run

...

### Results

...

### References

1. LoRA paper - [link](https://arxiv.org/abs/2106.09685)
2. LoRA-FA paper - [link](https://arxiv.org/abs/2308.03303)

### Authors

1. Sergey Karpukhin, [@hr3nk](https://github.com/shredder67)
2. Yulia Sergeeva
3. Pavel Bartenev
4. Pavel Tikhomirov, [@ocenandor](https://github.com/ocenandor)
5. Maksim Komiakov, [@kommaks](https://github.com/kommaks)
