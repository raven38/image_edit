# python 3.7
"""Demo."""

import os
import tempfile
from io import BytesIO

import numpy as np
import torch
import streamlit as st
import SessionState
from PIL import Image

from sefa.models import parse_gan_type
from utils import to_tensor, postprocess, load_generator, factorize_weight
from align_images import align_image
from encode_image import optimize_style

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model(model_name):
    """Gets model by name."""
    return load_generator(model_name)


@st.cache(allow_output_mutation=True, show_spinner=False)
def factorize_model(model, layer_idx):
    """Factorizes semantics from target layers of the given model."""
    return factorize_weight(model, layer_idx)


def sample(model, gan_type, num=1, codes=None):
    """Samples latent codes."""
    postf = lambda x: x
    if codes is None:
        postf = lambda x: x.detach().cpu().numpy()
        codes = torch.randn(num, model.z_space_dim).cuda()

    if gan_type == 'pggan':
        codes = model.layer0.pixel_norm(codes)
    elif gan_type == 'stylegan':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.7,
                                 trunc_layers=8)
    elif gan_type == 'stylegan2':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.5,
                                 trunc_layers=18)
    return postf(codes)

@st.cache(allow_output_mutation=True, show_spinner=False)
def synthesize(model, gan_type, code):
    """Synthesizes an image with the give code."""
    if gan_type == 'pggan':
        image = model(to_tensor(code))['image']
    elif gan_type in ['stylegan', 'stylegan2']:
        image = model.synthesis(to_tensor(code))['image']
    image = postprocess(image)[0]
    return image


@st.cache(allow_output_mutation=True, show_spinner=False)
def make_aligned_image(img_file_buffer, aligned_image_path):
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(img_file_buffer.name)[1]) as tname:
        Image.open(BytesIO(img_file_buffer.getvalue())).save(tname.name)
        align_image(tname.name, aligned_image_path)

def main():
    """Main function (loop for StreamLit)."""
    st.title('Closed-Form Factorization of Latent Semantics in GANs')
    st.sidebar.title('Options')
    reset = st.sidebar.button('Reset')

    model_name = st.sidebar.selectbox(
        'Model to Interpret',
        ['stylegan_animeportrait512',
         'stylegan2_ffhq1024',
         'stylegan_animeface512', 'stylegan_car512', 'stylegan_cat256',
         'stylegan_ffhq1024', 'stylegan_celebahq1024',
         'stylegan_celeba_partial256', 'stylegan_ffhq256', 'stylegan_ffhq512',
         'pggan_celebahq1024'])

    model = get_model(model_name)
    gan_type = parse_gan_type(model)
    layer_idx = st.sidebar.selectbox(
        'Layers to Interpret',
        ['all', '0-1', '2-5', '6-13'])
    layers, boundaries, eigen_values = factorize_model(model, layer_idx)

    num_semantics = st.sidebar.number_input(
        'Number of semantics', value=10, min_value=0, max_value=None, step=1)
    steps = {sem_idx: 0 for sem_idx in range(num_semantics)}
    if gan_type == 'pggan':
        max_step = 5.0
    elif gan_type == 'stylegan':
        max_step = 2.0
    elif gan_type == 'stylegan2':
        max_step = 15.0
    for sem_idx in steps:
        eigen_value = eigen_values[sem_idx]
        steps[sem_idx] = st.sidebar.slider(
            f'Semantic {sem_idx:03d} (eigen value: {eigen_value:.3f})',
            value=0.0,
            min_value=-max_step,
            max_value=max_step,
            step=0.04 * max_step if not reset else 0.0)

    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    st.text("After update image, please push the buttun named 'Optimize'.")
    image_placeholder = st.empty()
    button_placeholder = st.empty()
    button_opt_placeholder = st.empty()
    st.text("Pelease wait a few minutes until finishing optimization of latent vector.")
    st.text("You can repush optimize button if you hate an optimized latent vector.")

    aligned_image_path = 'align.png'
    if img_file_buffer is not None:
        make_aligned_image(img_file_buffer, aligned_image_path)


    try:
        base_codes = np.load(f'latent_codes/{model_name}_latents.npy')
    except FileNotFoundError:
        base_codes = sample(model, gan_type)

    state = SessionState.get(model_name=model_name,
                             code_idx=0,
                             codes=base_codes[0:1])
    if state.model_name != model_name:
        state.model_name = model_name
        state.code_idx = 0
        state.codes = base_codes[0:1]

    if button_opt_placeholder.button('Optimize', key=0):
        iteration = 500
        state.codes = optimize_style(aligned_image_path, model, model_name, gan_type, base_codes, iteration)

    if button_placeholder.button('Random', key=0):
        state.code_idx += 1
        if state.code_idx < base_codes.shape[0]:
            state.codes = base_codes[state.code_idx][np.newaxis]
        else:
            state.codes = sample(model, gan_type)

    code = state.codes.copy()
    for sem_idx, step in steps.items():
        if gan_type == 'pggan':
            code += boundaries[sem_idx:sem_idx + 1] * step
        elif gan_type in ['stylegan', 'stylegan2']:
            code[:, layers, :] += boundaries[sem_idx:sem_idx + 1] * step
    image = synthesize(model, gan_type, code)
    image_placeholder.image(image / 255.0)


if __name__ == '__main__':
    main()
