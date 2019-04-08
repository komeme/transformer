from typing import Dict

from torch import nn
from torch import Tensor

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits


class LstmTagger(Model):
    def __init__(
            self,
            word_embeddings: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            vocab: Vocabulary
    ):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2tag = nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size('labels')
        )

        self.accuracy = CategoricalAccuracy()

    def forward(
            self,
            sentence: Dict[str, Tensor],
            labels: Tensor = None
    ) -> Dict[str, Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
