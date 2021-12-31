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
matplotlib
scikit-learn
tqdm            # not mandatory but recommended
tensorboard     # not mandatory but recommended
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
python train.py --network_name='efficientnet_b0' --dataset_dir='cifar10' --epochs=5 --lr=0.1 \
--auto_mean_std --store_weights --store_loss_acc_log --store_logits \
--store_confusion_matrix --tag='your_experiment_name'
~~~

3. Run **test.py** for test. The below is an example. See **src/my_utils/parser.py** for details.
~~~ME
python test.py --network_name='efficientnet_b0' --dataset_dir='cifar10' \
--auto_mean_std --store_logits --store_confusion_matrix \
--checkpoint='pretrained_model_weights.pt'
~~~


### Contribution
🐛 If you find any bugs or have opinions for further improvements, feel free to contact us (yechankim@gm.gist.ac.kr or maestr.oh@gm.gist.ac.kr). All contributions are welcome.


### Reference
1. https://github.com/weiaicunzai/pytorch-cifar100
2. https://github.com/peterliht/knowledge-distillation-pytorch
3. https://medium.com/@djin31/how-to-plot-wholesome-confusion-matrix-40134fd402a8 (Confusion Matrix)
