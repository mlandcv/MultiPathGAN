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

##
