# DeepCU-IJCAI19
DeepCU: Integrating Both Common and Unique Latent Information for Multimodal Sentiment Analysis, IJCAI-19


![alt text](https://github.com/sverma88/DeepCU-IJCAI19/blob/master/figures/DeepCU.jpg)

## Citation
When using this code, or the ideas of DeepCU, please cite the following paper

    @INPROCEEDINGS{Verma0ZL19ijcai19,
     author = {Sunny Verma and Chen Wang and Liming Zhu and Wei Liu},
     title = DeepCU: Integrating both Common and Unique Latent Information for Multimodal Sentiment Analysis},
     booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI} Macao},
     pages     = {3627--3634},
     year      = {2019},
     }
     

### Dependencies / Notes
DeepCU is written in python3 with some code fragments copied from DCGAN implementation from Carpedm20.
  - The code is developed with Python 3.6 and TensorFlow 1.12.0 (with GPU support) on Linux
  - For reasons of my convenience, `data_dir` is required to be `data_dir = ../../data` -- errors might pop-up when other directories are used.
  - The experiments (main.py) - loads pretrained model and executes test (i.e. prediction with our trained model). If you wish to train on  your own data, please edit as deep_cu.train(FLAGS).
  
# Future research (ideas)
- Better Fusion scheme for utilizing both common and unique latent information
- Utilize Sequence information for sentiment prediction
- Cross Data Generelaization Performance


Please contact either Sunny Verma or Wei Liu at t firstname.lastname@uts.edu.au if you're interested to collaborate on this!

