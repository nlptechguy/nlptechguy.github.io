### AI frameworks and libraries
- Keras
- TensorFlow
- Scikit-learn
- Pandas
- NLTK 
- Seq2seq Model
- Word2vec
- LSTM, GRU, BERT
- N-Gram Model, TF-IDF
- NumPy
- Rasa NLU / Rasa X
- Spacy
- Deeppavlov


### Chatbot Platforms
- Flow xo
- Chatfuel
- IBM Watson
- Wit.ai
- Dialogflow
- Lex
- Microsoft Bot Platform / Azure / Luis / QnAMaker
- Recast.AI


Add simple sentiment analysis to Rasa

```
import logging
from typing import Any, Text, Dict, List, Type
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITIES,
)
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)
class SentimentAnalyzer(EntityExtractorMixin,GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        return []

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["nltk"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            "languages": [],
            "entity_name" : "sentiment",
            "extractor_name" : "NLTKVaderLexiconSentimentAnalyzer"
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.entity_name = config.get("entity_name")
        self.extractor_name = config.get("extractor_name")
        self.languages = config.get("languages")
        self.settings = {}
        
    def train(self, training_data: TrainingData) -> Resource:
        pass

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)


    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            self._set_entities(message)
        return messages

    def _set_entities(self, message: Message, **kwargs: Any) -> None:
        matches = self._extract_entities(message)
        message.set(ENTITIES, message.get(ENTITIES, []) + matches, add_to_output=True)

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        message_text =  message.get(TEXT)
        sentiments = []
        # process for english
        if "en" in self.languages:
            sid = SentimentIntensityAnalyzer()
            res = sid.polarity_scores(message_text)
            print('Sentiment score',res)
            sentiments.append({
                    ENTITY_ATTRIBUTE_TYPE: self.entity_name,
                    ENTITY_ATTRIBUTE_START: None,
                    ENTITY_ATTRIBUTE_END: None,
                    ENTITY_ATTRIBUTE_VALUE: res,
                    "extractor": self.extractor_name,
                })
        
        return sentiments

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass

```

Add pos tagging class in Rasa

```
from typing import Any, Text, Dict, List, Type, TypedDict, Optional
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITIES,
)

# import nltk dependencies
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import RegexpTokenizer

class Ngram(TypedDict):
    token: str
    pos: Optional[str]

Ngrams = Dict[str, List[Ngram]]

def tokenize(s: str):
    return RegexpTokenizer(r"\w+").tokenize(s)

def parse_tokens(tokens, max_length, pos=True) -> List[Ngram]:
    parsed: List[Ngram] = []
    for tagged in nltk.pos_tag(tokens):
        ngram: Ngram = {
            "token": tagged[0],
            "pos": tagged[1] if pos else None
        }
        parsed.append(ngram)
    return parsed

def find_ngrams_nltk(text:str, max_length: int) -> Ngrams:
    tokens = tokenize(text)
    ngrams: Ngrams = {"n1": parse_tokens(tokens, max_length)}
    for n in range(2, max_length + 1):
        ngram_list = []
        # If there are enough tokens find the ngrams of them
        if len(tokens) >= n:
            for ngram in list(nltk.ngrams(tokens, n)):
                ngram_list.append(" ".join(ngram))

        ngrams["n" + str(n)] = parse_tokens(ngram_list, max_length, False)
    return ngrams

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)
class PartOfSpeechTagging(EntityExtractorMixin,GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        return []

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["nltk"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            "languages": [],
            "max_length" : 1,
            "entity_name" : "pos_tagging",
            "extractor_name" : "NLTKPartOfSpeechTagging"
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.entity_name = config.get("entity_name")
        self.extractor_name = config.get("extractor_name")
        self.max_length = config.get("max_length")
        self.languages = config.get("languages")
        self.settings = {}
        
    def train(self, training_data: TrainingData) -> Resource:
        pass

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)


    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            self._set_entities(message)
        return messages

    def _set_entities(self, message: Message, **kwargs: Any) -> None:
        matches = self._extract_entities(message)
        message.set(ENTITIES, message.get(ENTITIES, []) + matches, add_to_output=True)

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        message_text =  message.get(TEXT)
        pos_tagging = []
        # process for english
        if "en" in self.languages:
            pos_tagging.append({
                ENTITY_ATTRIBUTE_TYPE: self.entity_name,
                ENTITY_ATTRIBUTE_START: None,
                ENTITY_ATTRIBUTE_END: None,
                ENTITY_ATTRIBUTE_VALUE: find_ngrams_nltk(message_text,self.max_length),
                "extractor": self.extractor_name,
            })
        return  pos_tagging

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
```
