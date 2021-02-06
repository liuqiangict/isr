# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "2.11.0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging

# Configurations
from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, AutoConfig
from .configuration_bart import BartConfig
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
from .configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from .configuration_encoder_decoder import EncoderDecoderConfig
from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
from .configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from .configuration_marian import MarianConfig
from .configuration_mmbt import MMBTConfig
from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from .configuration_reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from .configuration_utils import PretrainedConfig
from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
from .configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig


is_torch_available=True

# Modeling
if is_torch_available:
    from .modeling_utils import PreTrainedModel, prune_layer, Conv1D, top_k_top_p_filtering, apply_chunking_to_forward
    from .modeling_auto import (
        AutoModel,
        AutoModelForPreTraining,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelWithLMHead,
        AutoModelForTokenClassification,
        AutoModelForMultipleChoice,
        MODEL_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_WITH_LM_HEAD_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    )

    from .modeling_bert import (
        BertPreTrainedModel,
        BertModel,
        BertForPreTraining,
        BertForMaskedLM,
        BertForNextSentencePrediction,
        BertForSequenceClassification,
        BertForMultipleChoice,
        BertForTokenClassification,
        BertForQuestionAnswering,
        load_tf_weights_in_bert,
        BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        BertLayer,
    )
    from .modeling_openai import (
        OpenAIGPTPreTrainedModel,
        OpenAIGPTModel,
        OpenAIGPTLMHeadModel,
        OpenAIGPTDoubleHeadsModel,
        load_tf_weights_in_openai_gpt,
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_transfo_xl import (
        TransfoXLPreTrainedModel,
        TransfoXLModel,
        TransfoXLLMHeadModel,
        AdaptiveEmbedding,
        load_tf_weights_in_transfo_xl,
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_gpt2 import (
        GPT2PreTrainedModel,
        GPT2Model,
        GPT2LMHeadModel,
        GPT2DoubleHeadsModel,
        load_tf_weights_in_gpt2,
        GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_ctrl import CTRLPreTrainedModel, CTRLModel, CTRLLMHeadModel, CTRL_PRETRAINED_MODEL_ARCHIVE_LIST
    from .modeling_xlnet import (
        XLNetPreTrainedModel,
        XLNetModel,
        XLNetLMHeadModel,
        XLNetForSequenceClassification,
        XLNetForTokenClassification,
        XLNetForMultipleChoice,
        XLNetForQuestionAnsweringSimple,
        XLNetForQuestionAnswering,
        load_tf_weights_in_xlnet,
        XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_xlm import (
        XLMPreTrainedModel,
        XLMModel,
        XLMWithLMHeadModel,
        XLMForSequenceClassification,
        XLMForTokenClassification,
        XLMForQuestionAnswering,
        XLMForQuestionAnsweringSimple,
        XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_bart import (
        BartForSequenceClassification,
        BartModel,
        BartForConditionalGeneration,
        BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_marian import MarianMTModel
    from .modeling_roberta import (
        RobertaForMaskedLM,
        RobertaModel,
        RobertaForSequenceClassification,
        RobertaForMultipleChoice,
        RobertaForTokenClassification,
        RobertaForQuestionAnswering,
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_distilbert import (
        DistilBertPreTrainedModel,
        DistilBertForMaskedLM,
        DistilBertModel,
        DistilBertForSequenceClassification,
        DistilBertForQuestionAnswering,
        DistilBertForTokenClassification,
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_camembert import (
        CamembertForMaskedLM,
        CamembertModel,
        CamembertForSequenceClassification,
        CamembertForMultipleChoice,
        CamembertForTokenClassification,
        CamembertForQuestionAnswering,
        CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_encoder_decoder import EncoderDecoderModel
    from .modeling_t5 import (
        T5PreTrainedModel,
        T5Model,
        T5ForConditionalGeneration,
        load_tf_weights_in_t5,
        T5_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_albert import (
        AlbertPreTrainedModel,
        AlbertModel,
        AlbertForPreTraining,
        AlbertForMaskedLM,
        AlbertForSequenceClassification,
        AlbertForQuestionAnswering,
        AlbertForTokenClassification,
        load_tf_weights_in_albert,
        ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_xlm_roberta import (
        XLMRobertaForMaskedLM,
        XLMRobertaModel,
        XLMRobertaForMultipleChoice,
        XLMRobertaForSequenceClassification,
        XLMRobertaForTokenClassification,
        XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_mmbt import ModalEmbeddings, MMBTModel, MMBTForClassification

    from .modeling_flaubert import (
        FlaubertModel,
        FlaubertWithLMHeadModel,
        FlaubertForSequenceClassification,
        FlaubertForQuestionAnswering,
        FlaubertForQuestionAnsweringSimple,
        FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

    from .modeling_electra import (
        ElectraForPreTraining,
        ElectraForMaskedLM,
        ElectraForTokenClassification,
        ElectraPreTrainedModel,
        ElectraForSequenceClassification,
        ElectraModel,
        load_tf_weights_in_electra,
        ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

    from .modeling_reformer import (
        ReformerAttention,
        ReformerLayer,
        ReformerModel,
        ReformerModelWithLMHead,
        REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

    from .modeling_longformer import (
        LongformerModel,
        LongformerForMaskedLM,
        LongformerForSequenceClassification,
        LongformerForMultipleChoice,
        LongformerForTokenClassification,
        LongformerForQuestionAnswering,
        LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

    from .modeling_sparse import (
        SparseModel,
        SparseForTokenClassification,
    )

