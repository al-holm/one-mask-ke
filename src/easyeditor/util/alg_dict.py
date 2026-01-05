from ..models.rome import apply_rome_to_model
from ..models.memit import apply_memit_to_model
from ..models.kn import apply_kn_to_model
from ..models.mend import (
    MendRewriteExecutor,
    MendMultimodalRewriteExecutor,
    MendPerRewriteExecutor,
)
from ..models.ft import apply_ft_to_model
from ..models.dinm import apply_dinm_to_model
from ..models.serac import (
    SeracRewriteExecutor,
    SeracMultimodalRewriteExecutor,
)
from ..dataset import (
    ZsreDataset,
    CounterFactDataset,
    CaptionDataset,
    VQADataset,
    PersonalityDataset,
    SafetyDataset,
)
from ..models.ike import (
    apply_ike_to_model,
    apply_ike_to_multimodal_model,
    apply_ike_to_per_model,
)
from ..models.ft_api import apply_ft_api_to_model
from ..models.qlora import apply_qlora_to_model
from ..models.lora import (
    apply_lora_to_model,
    apply_lora_to_multimodal_model,
)
from ..models.grace import (
    apply_grace_to_model,
    apply_grace_to_multimodal_model,
)
from ..models.pmet import apply_pmet_to_model
from ..models.melo import apply_melo_to_model
from ..models.wise import (
    apply_wise_to_model,
    apply_wise_to_multimodal_model,
)
from ..models.r_rome import apply_r_rome_to_model
from ..models.emmet import apply_emmet_to_model
from ..models.alphaedit import apply_AlphaEdit_to_model
from ..models.deepedit_api import apply_deepedit_api_to_model
from ..models.dpo import apply_dpo_to_model

ALG_DICT = {
    "ROME": apply_rome_to_model,
    "MEMIT": apply_memit_to_model,
    "FT": apply_ft_to_model,
    "DINM": apply_dinm_to_model,
    "KN": apply_kn_to_model,
    "MEND": MendRewriteExecutor().apply_to_model,
    "SERAC": SeracRewriteExecutor().apply_to_model,
    "IKE": apply_ike_to_model,
    "FT-Api": apply_ft_api_to_model,
    "QLoRA": apply_qlora_to_model,
    "LoRA": apply_lora_to_model,
    "DPO": apply_dpo_to_model,
    "GRACE": apply_grace_to_model,
    "PMET": apply_pmet_to_model,
    "MELO": apply_melo_to_model,
    "WISE": apply_wise_to_model,
    "R-ROME": apply_r_rome_to_model,
    "EMMET": apply_emmet_to_model,
    "AlphaEdit": apply_AlphaEdit_to_model,
    "DeepEdit-Api": apply_deepedit_api_to_model,
}

ALG_MULTIMODAL_DICT = {
    "MEND": MendMultimodalRewriteExecutor().apply_to_model,
    "SERAC": SeracMultimodalRewriteExecutor().apply_to_model,
    "SERAC_MULTI": SeracMultimodalRewriteExecutor().apply_to_model,
    "IKE": apply_ike_to_multimodal_model,
    "LoRA": apply_lora_to_multimodal_model,
    "WISE": apply_wise_to_multimodal_model,
    "GRACE": apply_grace_to_multimodal_model,
}

PER_ALG_DICT = {
    "IKE": apply_ike_to_per_model,
    "MEND": MendPerRewriteExecutor().apply_to_model,
}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset,
}

MULTIMODAL_DS_DICT = {
    "caption": CaptionDataset,
    "vqa": VQADataset,
}

PER_DS_DICT = {"personalityEdit": PersonalityDataset}
Safety_DS_DICT = {"safeEdit": SafetyDataset}
