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
