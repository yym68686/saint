import os
import logging
from dotenv import load_dotenv
load_dotenv()

# Configure logger to provide feedback on which module is loaded.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Get the desired SAE architecture from an environment variable. Default to 'topk'.
SAE_ARCHITECTURE = os.environ.get("SAE_ARCHITECTURE", "topk").lower()

if SAE_ARCHITECTURE == "dense":
    logging.info("Using 'dense' SAE architecture from 'sae_exp11_dense'.")
    from sae_exp11_dense import load_sae_model, TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "batchtopk":
    logging.info("Using 'batchtopk' SAE architecture from 'sae_batchtopk'.")
    from sae_batchtopk import load_sae_model, BatchTopKSparseAutoencoder as TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "sigreg":
    logging.info("Using 'sigreg' SAE architecture from 'sae_sigreg'.")
    from sae_sigreg import load_sae_model, TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "repreg":
    logging.info("Using 'repreg' SAE architecture from 'sae'.")
    from sae import load_sae_model, TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "sigregrepreg":
    logging.info("Using 'sigregrepreg' SAE architecture from 'sae'.")
    from sae import load_sae_model, TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "kernel":
    logging.info("Using 'kernel' SAE architecture from 'sae'.")
    from sae import load_sae_model, TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "denseimq":
    logging.info("Using 'denseimq' SAE architecture from 'sae_exp11_dense'.")
    from sae_exp11_dense import load_sae_model, TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "denserepreg":
    logging.info("Using 'denserepreg' SAE architecture from 'sae_dense_repreg'.")
    from sae_dense_repreg import load_sae_model, TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "imq":
    logging.info("Using 'imq' SAE architecture from 'sae'.")
    from sae import load_sae_model, TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "relu":
    logging.info("Using 'relu' SAE architecture from 'sae_relu'.")
    from sae_relu import load_sae_model, ReluSAE as TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "jumprelu":
    logging.info("Using 'jumprelu' SAE architecture from 'sae_jumprelu'.")
    from sae_jumprelu import load_sae_model, JumpReLUSAE as TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "gatedsae":
    logging.info("Using 'gatedsae' SAE architecture from 'sae_gatedsae'.")
    from sae_gatedsae import load_sae_model, GatedSAE as TopKSparseAutoencoder
elif SAE_ARCHITECTURE == "topk":
    logging.info("Using 'topk' SAE architecture from 'sae'.")
    from sae import load_sae_model, TopKSparseAutoencoder
else:
    # If an unknown value is provided, raise an error to prevent unexpected behavior.
    raise ImportError(
        f"Unknown SAE_ARCHITECTURE: '{SAE_ARCHITECTURE}'. "
        "Please set the environment variable to one of 'topk', 'dense', 'kernel', 'sigreg', 'repreg', 'sigregrepreg', 'imq', 'denseimq', 'batchtopk', 'relu', 'jumprelu' or 'gatedsae'."
    )

# Expose the loaded symbols for other modules to import.
__all__ = ["load_sae_model", "TopKSparseAutoencoder"]
