#!/usr/bin/env python
import os
import sys
from keras.models import load_model
from icecream import ic

# seisbench denoiser
import seisbench.models as sbm

SEIS_DAE = "/Users/marc/github/seisDAE"
sys.path.append(SEIS_DAE)
from denoiser.denoise_utils import denoising_stream

MODEL = os.path.join(SEIS_DAE, "Models", "gr_mixed_stft.h5")
CONFIG = os.path.join(SEIS_DAE, "config", "gr_mixed_stft.config")


def denoise_config_check():
    # check if MODEL and CONFIG are defined and exist
    if not os.path.isfile(MODEL):
        logger.error("Deep denoise model file '%s' not found !", MODEL)
        return False
    if not os.path.isfile(CONFIG):
        logger.error("Deep denoise config file '%s' not found !", CONFIG)
        return False
    return True


def denoise_stream(stream, model_name=None, preprocess=True):
    print(model_name)

    assert model_name in (
        "dae",
        "original",
        "urban",
    ), "Model name must be 'dae', 'original' or 'urban'"

    st = stream.copy()

    if preprocess:
        st.detrend(type="demean")
        st.detrend(type="linear")
        st.taper(max_percentage=0.05, type="cosine", side="both")

    if model_name == "dae":
        try:
            model_dae = load_model(MODEL)
        except ValueError:
            model_dae = load_model(MODEL, compile=False)
        st_denoised, st_noise = denoising_stream(
            stream=st, config_filename=CONFIG, loaded_model=model_dae, parallel=True
        )
    else:
        denoise_model = sbm.DeepDenoiser.from_pretrained(model_name)
        st_denoised = denoise_model.annotate(st)

    # copy coordinates and response to denoised traces
    for tr in st_denoised:
        if model_name in ['original', 'urban']:
            # remove the 'DeepDenoiser_' prefix
            tr.stats.channel = tr.stats.channel.split("_")[1]
            
        # Find corresponding trace 
        mytrace = st.select(id=tr.id)
        assert mytrace, f"Something when wrong when finding {tr.id}"

        tr.stats.coordinates = mytrace[0].stats.coordinates
        tr.stats.response = mytrace[0].stats.response

    return st_denoised
