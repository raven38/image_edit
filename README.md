# Neural Image Editor

The repository contains GAN-based image editing.

Demo is avaiable on [Colab](https://colab.research.google.com/github/raven38/image_edit/blob/master/image_edit_demo.ipynb).

![demo](./demo.gif)
The [demo video](https://drive.google.com/file/d/1OfDFguZtf4cNcJ32FRNmeW1Aq3IDNN2S/view?usp=sharing) (long version) is available. 

## Interface

We also provide an interface for interactive editing based on StreamLit. This interface can be locally launched with 
```
CUDA_VISIBLE_DEVICES=0 streamlit run interface.py
```

*NOTE:* User can upload your own image though the file upload
dialog. Please wait a few minutes for finishing the optimization of
latent code that suits for the uploaded image.


## Code

This demo is based on two algorithms: [Image2StyleGAN](https://arxiv.org/abs/1904.03189) and [SeFa](https://genforce.github.io/sefa/).
It borrows some codes from [Image2StyleGAN](https://github.com/pacifinapacific/StyleGAN_LatentEditor) and [Sefa](https://github.com/genforce/sefa) repositories.
