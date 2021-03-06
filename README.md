[![Python 3.8.12](https://img.shields.io/badge/python-3.8.12+-blue)](https://www.python.org/downloads/release/python-3812/)
[![torch](https://img.shields.io/badge/torch-1.10.0+-green)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.11.1+-green)](https://pytorch.org/)
[![datasets](https://img.shields.io/badge/datasets-1.17.0+-green)](https://huggingface.co/docs/datasets/index)
[![transformers](https://img.shields.io/badge/transformers-4.15.0+-green)](https://huggingface.co/docs/transformers/index)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20Hub-yellow)](https://huggingface.co/models?filter=keytotext)

# CycleTransformer

Both Text-to-Image translation and Image-to-Text translation have been an active area of research in the recent past [5,6,7]. Both tasks are difficult and interesting problems to solve: The Image-to-Text task demands that the generated caption will faithfully describe the image, while the Text-to-Image task demands that the generated image should be a faithful visual representation of the given text. Usually only one task is handled at a time, and the methods are tailored for extracting data from one domain and translating it to the other domain. 

Recently, some [2,3,4] took inspiration from CycleGAN's use of duality for unpaired data [1], by leveraging the cycle consistency duality for paired data of different domains, such as text and images. Inspired by those papers and recent advancements in deep learning and NLP, in this assignment we propose a novel architecture, CycleTransformer, to handle both Text-to-Image translation and Image-to-Text translation on paired data, using a unified architecture of transformers and CNNs and enforcing cycle consistency.

## Table of Contents
* [Requirements](#requirements)
* [Repository Structure](repository-structure) 
* [Usage Example](#usage-example)
* [Model](#model)
* [Team](#team)
* [Examples](#examples)
* [Refrences](#refrences)

## Requirements
The code was tested on python v3.8.12 with the following libraries:
| Library | Version |
| ------------- | ------------- |
| `datasets` | `1.17.0` |
| `matplotlib` | `3.4.3` |
| `numpy` | `1.21.3` |
| `pillow` | `8.4.0` |
| `pytorch` | `1.10.0+cu111` |
| `pytorch-fid` | `0.2.1` |
| `rouge_score` | `0.0.4` |
| `scikit-image` | `0.18.3` |
| `scipy` | `1.7.1` |
| `torchvision` | `0.11.1+cu111` |
| `tqdm` | `4.63.0` |
| `transformers` | `4.15.0` |


We recommend using conda to deploy the project:  
```bash
git glone https://github.com/HilaManor/CycleTransformer.git && cd CycleTransformer
conda create --name CycleTransformer python=3.8.12 pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.1 numpy=1.21.3 scikit-image=0.18.3 matplotlib=3.4.3 scipy=1.7.1 pandas=1.3.4 pillow=8.4.0 tqdm -c pytorch -c conda-forge
conda activate CycleTransformer
pip install transformers==4.15.0 datasets==1.17.0 rouge_score==0.0.4 pytorch-fid==0.2.1
```
### IMPORTANT - Fixing Hugginface Bug

The transformers code we've been working with had a bug which didn't allow the use of tensors in the ViT feature extraction method. We had to fix this bug in the library's code to allow complete gradient flow (for the consistency cycle).  
This means that for our code to run, until the bug will fixed in the offical repo, you must fix it yourself before running the code.

To fix the bug you should edit `feature_extraction_utils.py` located in `<python_base_folder>/site-packages/transformers/`:  
line 144 (under the function `as_tensor(value)`, declared in line 142 of transformersv4.15.0):  
add:
```python 
elif isinstance(value, (list, torch.Tensor)):
    return torch.stack(value)
```


## Repository Structure 
```
????????? code - the code for training the CycleTransformer model
????????? config - configurations for the CycleTransformer model
```

## Usage Example
### Training the model 

```bash
python main.py --epochs <training_epochs> --val_epochs <validation_every_x_epochs> --config <path_to_yaml_file> [--baseline]
```

At each validation epoch the validation loss is shown and some images and captions are created from the validation split.  
If the optional `--baseline` is given, will train the baseline models instead.  
Use `--help` for more information on the parameters.

### Generating Images and Captions

```bash
python main.py --out_dir <path_to_trained_model_dir> [--text <optional_text_prompt>] [--img_path <optional_image_path>] [--amount <amount_of_images_to_generate>]
```

Generates the images of the test split and generate captions for them, while comparing to the ground truths.  
If the optional `--text` is given, will generate images from that text. The amount of generated images is given by `--amount`.  
If the optional `--img_path` is given, will generate a text caption for the given image.  
Use `--help` for more information on the parameters.

### Pretrained Weights and Organized Dataset

Pretrained weights for the cycle consistent model are hosted [here](https://technionmail-my.sharepoint.com/:u:/g/personal/matankleiner_campus_technion_ac_il/EdACN8TSdyVLjoE_mbmk7yYBkUpziemLWeIeFVKfgA4DJQ?e=u7wYlb) and pretrained weights for the baseline models are hosted [here](https://technionmail-my.sharepoint.com/:u:/g/personal/matankleiner_campus_technion_ac_il/EdACN8TSdyVLjoE_mbmk7yYBkUpziemLWeIeFVKfgA4DJQ?e=hRcFE1).

The dataset, organized as we used it (without classes splits) is available for simplicity [here](https://technionmail-my.sharepoint.com/:u:/g/personal/matankleiner_campus_technion_ac_il/EUyOIAu6vsdJlXGM0RSHveoBcdoD0uTVB8JpbjGz840nGw?e=un1gdG).

## Model 
CycleTransformer model is comprised of Text-to-Image and Image-to-Text parts.  
The Text-to-Image model is comprised of distill BERT for text embedding. We concatenate a random noise vector sampled from the standard normal distributed to this embedding and then feed it to an image generator model. The Text-to-Image model is trained using perceptaul and reconstruction losses.  
The Image-to-text model is an encoder decoder structure composed of distill DeiT model for features extractor and a GPT2 for text generation. This model is trained using language modelling loss.  
Read our short paper for more detailes about the model. 

![model](https://user-images.githubusercontent.com/63591190/157965275-6a900647-f0ad-4421-b799-df3a00d835ed.png)

## Team
Hila Manor and Matan Kleiner

## Examples
![image](https://user-images.githubusercontent.com/53814901/158409123-67fb7b88-79d9-4626-ae1b-73e1b0ac8fd5.png)

Comparison between the baseline models and the cycle consistent model on more generation results, for the Text-to-Image task and the Image-to-Test task. The baseline models generations are more varied in shape and display more coherent colors. The generated sentences are also more diverse in the language used and the different elements of the flower are described with greater detail.

![image](https://user-images.githubusercontent.com/53814901/158409282-7a2834a8-cbc4-482e-9db9-0c61f41522c6.png)

Comparison between the results of the baseline and the cycle-consistent Image-to-Text models, for custom images on the Image-to-Text task. All the image were found online and are under creative commons license. The first three flowers (Gilboa Iris, Nazareth Iris and Mountain Tulip, respectively) are not part of the original dataset flower species. The last flower (Daffodil) is a specie present in the original dataset. 
    The caption created by the baseline model are more accurate and do not include colors that don't appear in the input image. The cycle consistent model also creates good captions but in the case of the Nazareth Iris and the Daffodil, it mentions that the flower's color is purple which is not the case.



![image](https://user-images.githubusercontent.com/53814901/158409298-f33afed7-278a-4909-80c5-aa333385b142.png)

Comparison between the results of the baseline and the cycle-consistent Text-to-Image models, for custom text prompts on the Text-to-Image task. The first two sentences are simple, describing a single flower with one prominent color. In both cases, both models create a blob of the specified color in the center of the image, where the baseline model's blob's shape is a bit more flower-like. The next two sentences are more complicated, one of them describes more than one flower and the other describes 3 different-colored parts of the flower. The first sentence created a similar image response for both models. The seconds sentence caused the baseline model to generate a blob that merges two of the described colors whereas the cycle consistent model generated a colorful blob, without relating it to the mentioned colors. The last two sentences are the most complex ones. Both models generate similar results, while the the baseline model's results are a bit more pleasing to the eye.


## References
1. Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. 2017. [Unpaired image-to-image translation using cycle-consistent adversarial networks](https://arxiv.org/abs/1703.10593), In *Proceedings of the IEEE international conference on computer vision*, pages 2223???2232. 
2. Mohammad R. Alam, Nicole A. Isoda, Mitch C. Manzanares, Anthony C. Delgado, and Antonius F. Panggabean. 2021. [TextCycleGAN: cyclical-generative adversarial networks for image captioning](https://spie.org/Publications/Proceedings/Paper/10.1117/12.2585549), In *Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications* III, volume 11746, pages 213 ??? 220. International Society for Optics and Photonics, SPIE
3. Satya Krishna Gorti and Jeremy Ma. 2018. [Text-to-image-to-text translation using cycle consistent adversarial networks](https://arxiv.org/abs/1808.04538), *arXiv preprint*, arXiv:1808.04538
4. Keisuke Hagiwara, Yusuke Mukuta, and Tatsuya Harada. 2019. [End-to-end learning using cycle consistency for image-to-caption transformations](https://arxiv.org/abs/1903.10118), *arXiv preprint*, arXiv:1903.10118
5. Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. 2020. [Oscar: Object semantics aligned pre-training for vision-language tasks](https://arxiv.org/abs/2004.06165), In *European Conference on Computer Vision*, pages 121???137. Springer.
6. Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2021. [Glide: Towards photorealistic image generation and editing with text-guided diffusion models](https://arxiv.org/abs/2112.10741), *arXiv preprint*, arXiv:2112.10741
7. Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. 2021. [Zero-shot text-to-image generation](https://arxiv.org/abs/2102.12092), In *International Conference on Machine Learning*, pages 8821???8831. PMLR.
