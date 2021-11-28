# Knowledge-Distillation
## Is it possible to effectively train student models using logits with intervention? (Tentative Title) 
[Yechan Kim*](https://github.com/unique-chan) and [Junggyun Oh*](https://github.com/Dodant)
(* denotes equal contribution.)

🚧 Under Construction! (Do not fork this repository yet!)

### This repository contains:
- Python3 / Pytorch code for response-based knowledge distillation


### Prerequisites
- See `requirements.txt` for details.
~~~ME
torch
torchvision
tqdm            # not mandatory
tensorboard     # not mandatory
~~~


### How to use
1. The directory structure of your dataset should be as follows.
~~~
|—— 📁 your_own_dataset
	|—— 📁 train
		|—— 📁 class_1
			|—— 🖼️ 1.jpg
			|—— ...
		|—— 📁 class_2 
			|—— 🖼️ ...
	|—— 📁 valid
		|—— 📁 class_1
		|—— 📁 ... 
	|—— 📁 test
		|—— 📁 class_1
		|—— 📁 ... 
~~~

2. Run **train.py** for training. The below is an example. See **src/my_utils/parser.py** for details.
~~~ME
python train.py --network_name='efficientnet_b0' --dataset_dir='./cifar10' --epochs=1 --lr=0.1 --compute_mean_std --store --tag='yechan-experiment1'
~~~


### Contribution
If you find any bugs or have opinions for further improvements, feel free to contact us (yechankim@gm.gist.ac.kr or maestr.oh@gm.gist.ac.kr). All contributions are welcome.


### Reference
1. https://github.com/weiaicunzai/pytorch-cifar100
2. https://github.com/peterliht/knowledge-distillation-pytorch
