from typing import Dict, List, Iterator

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField


class PosDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {'sentence': sentence_field}

        if tags is not None:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields['labels'] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split('/') for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)