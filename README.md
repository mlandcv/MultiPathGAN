# MultiPathGAN
This repository provides the official PyTorch implementation of the following paper:
> **MultiPathGAN: Structure Preserving Stain Normalization using Unsupervised Multi-domain Adversarial Network with Perception Loss**<br>
> [Haseeb Nazki](https://github.com/mlandcv)<sup>1</sup>, [Ognjen Arandjelović](https://risweb.st-andrews.ac.uk/portal/en/persons/oggie-arandelovic(fdd98ab1-564a-42a3-bf0c-fab7afbbd63c).html)<sup>1</sup>, [InHwa Um](https://risweb.st-andrews.ac.uk/portal/en/persons/in-hwa-um(0ac978a2-6ef8-4397-bc36-f920a77696a3).html)<sup>1</sup>, [David Harrison](https://risweb.st-andrews.ac.uk/portal/en/persons/david-james-harrison(6bb6c114-15d1-4b0d-9091-8ce3ce9c2c7d).html)<sup>1</sup><br/>
> <sup>1</sup>University of St-Andrews.<br/>
> https://arxiv.org/abs/1711.09020 <br>
>
> **Abstract:** *Histopathology relies on the analysis of microscopic tissue images to diagnose disease. A crucial part of tissue preparation is staining whereby a dye is used to make the salient tissue components more distinguishable. However, differences in laboratory protocols and scanning devices result in significant confounding appearance variation in the corresponding images. This variation increases both human error and the inter-rater variability, as well as hinders the performance of automatic or semi-automatic methods. In the present paper we introduce an unsupervised adversarial network to translate (and hence normalize) whole slide images across multiple data acquisition domains. Our key contributions are: (i) an adversarial architecture which learns across multiple domains with a single generator-discriminator network using an information flow branch which optimizes for perceptual loss, and (ii) the inclusion of an additional feature extraction network during training which guides the transformation network to keep all the structural features in the tissue image intact. We: (i) demonstrate the effectiveness of the proposed method firstly on H\&E slides of 120 cases of kidney cancer, as well as (ii) show the benefits of the approach on more general problems, such as flexible illumination based natural image enhancement and light source adaptation.*
> 
<p align="center"><img width="100%" src="image/image1new.jpg" /></p>

*Figure 1. Model Architecture* 
> 
<p align="center"><img width="100%" src="image/image3.jpg" /></p>

*Figure 2. Translation results between WSI patches from three different scanning devices* 
> 
<p align="center"><img width="100%" src="image/image9.jpg" /></p>

*Figure 1. Translating input to 5 color temperatures (MultiPathGAN trained on [VIDIT](https://github.com/majedelhelou/VIDIT))* 

## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)
* [TensorFlow 1.14+](https://www.tensorflow.org/) (optionally to use tensorboard)

## Prepare Dataset
#### 1. Split wsi patches into training and test sets (e.g., 90\%/10\% for training and test, respectively).  
#### 2. Save images in the format shown below:


    data
    └── WSI
        ├── train
        |   ├── Domain 1
        |   |   ├── a.jpg  (name doesn't matter)
        |   |   ├── b.jpg
        |   |   └── ...
        |   ├── Domain 2
        |   |   ├── c.jpg
        |   |   ├── d.jpg
        |   |   └── ...
        |   ...
        |
        └── test
            ├── Domain 1
            |   ├── e.jpg
            |   ├── f.jpg
            |   └── ...
            ├── Domain 2
            |   ├── g.jpg
            |   ├── i.jpg
            |   └── ...
            ...

## Training, testing and sampling MultiPathGAN on your dataset
To train on your own dataset, run the script provided below. 
```bash
# Train MultiPathGAN
python main.py --mode train
```
To test your trained network, run the script provided below.
```bash
# Test MultiPathGAN
python main.py --mode test 
```
To sample (translate test directory) to particular domain, run the script provided below.

# Sample MultiPathGAN
```bash
python main.py --mode test --which_domain 0
```
Change additional arguments, to suit your needs [here]().
