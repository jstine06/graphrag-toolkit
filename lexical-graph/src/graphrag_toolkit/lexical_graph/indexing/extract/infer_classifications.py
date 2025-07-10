# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import logging
import random
from typing import Sequence, List, Any, Optional

from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.indexing.extract.infer_config import OnExistingClassifications
from graphrag_toolkit.lexical_graph.indexing.extract.source_doc_parser import SourceDocParser
from graphrag_toolkit.lexical_graph.indexing.extract import ScopedValueStore, DEFAULT_SCOPE
from graphrag_toolkit.lexical_graph.indexing.constants import DEFAULT_ENTITY_CLASSIFICATIONS
from graphrag_toolkit.lexical_graph.indexing.prompts import DOMAIN_ENTITY_CLASSIFICATIONS_PROMPT
from graphrag_toolkit.lexical_graph.indexing.prompts import RANK_ENTITY_CLASSIFICATIONS_PROMPT

from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.bridge.pydantic import Field
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

DEFAULT_NUM_SAMPLES = 5
DEFAULT_NUM_ITERATIONS = 1
DEFAULT_NUM_CLASSIFICATIONS = 15

class InferClassifications(SourceDocParser):
    """
    This class is responsible for inferring domain-specific classifications for a set of
    textual nodes using a combination of sampling, prompt-based extraction via an LLM, and
    domain-adaptation strategies. It supports iterative refinement across multiple samples
    and customizable actions for handling pre-existing classifications.

    The class integrates classification parsing, chunk splitting, LLM-based prediction,
    and classification storage. It is designed to adaptively extend or replace existing
    classifications for textual input nodes based on analyzed data.

    Attributes:
        classification_store (ScopedValueStore): Storage for managing the scoped
            domain-specific classifications.
        classification_label (str): Label associated with the classifications to be
            inferred or modified.
        classification_scope (str): Scope for grouping the classifications within
            the classification store.
        num_samples (int): Number of chunks to sample for each iteration of the
            classification inference process.
        num_iterations (int): The number of times to iterate over sampling and
            classification inference.
        splitter (Optional[SentenceSplitter]): Responsible for splitting nodes into
            smaller chunks for processing, if provided.
        llm (Optional[LLMCache]): The LLM used for processing and generating domain-
            specific classifications based on the input data.
        prompt_template (str): Prompt template for LLM interaction, used to define
            how chunks are presented for classification analysis.
        default_classifications (List[str]): Default classifications to use if no
            domain-specific classifications could be inferred.
        merge_action (OnExistingClassifications): Defines the behavior for handling
            pre-existing classifications when processing new data.
    """
    classification_store:ScopedValueStore = Field(
        description='Classification store'
    )

    classification_label:str = Field(
        description='Classification label'
    )

    classification_scope:str = Field(
        description='Classification scope'
    )

    num_samples:int = Field(
        description='Number of chunks to sample per iteration'
    )

    num_iterations:int = Field(
        description='Number times to sample documents'
    )

    num_classifications:int = Field(
        description='Number of classifications to return'
    )

    splitter:Optional[SentenceSplitter] = Field(
        description='Chunk splitter'
    )

    llm: Optional[LLMCache] = Field(
        description='The LLM to use for extraction'
    )

    prompt_template:str = Field(
        description='Prompt template'
    )

    rank_prompt_template:str = Field(
        description='Prompt template'
    )

    default_classifications:List[str] = Field(
        'Default classifications'
    )

    merge_action:OnExistingClassifications = Field(
        'Action to take if there are existing classifications'
    )

    

    def __init__(self,
                 classification_store:ScopedValueStore,
                 classification_label:str,
                 classification_scope:Optional[str]=None,
                 num_samples:Optional[int]=None, 
                 num_iterations:Optional[int]=None,
                 num_classifications:Optional[int]=None,
                 splitter:Optional[SentenceSplitter]=None,
                 llm:Optional[LLMCacheType]=None,
                 prompt_template:Optional[str]=None,
                 rank_prompt_template:Optional[str]=None,
                 default_classifications:Optional[List[str]]=None,
                 merge_action:Optional[OnExistingClassifications]=None
            ):
        """
        Initializes an instance of the class with various parameters for classification and
        text processing. The initializer sets default values for optional parameters when
        none are provided and ensures that objects of specific types are created or verified
        before being utilized in the classification process.

        Args:
            classification_store: ScopedValueStore that serves as the storage mechanism for
                classifications.
            classification_label: String representing the label to be used for classification tagging.
            classification_scope: Optional string specifying the scope of classifications to be
                applied.
            num_samples: Optional integer specifying the number of samples to consider during
                classification.
            num_iterations: Optional integer indicating the number of iterations for the
                classification process.
            splitter: Optional SentenceSplitter object to handle sentence splitting operations.
            llm: Optional LLMCacheType specifying the language model to be used. Defaults to an
                internal configuration if not provided.
            prompt_template: Optional string containing the template for prompt generation in
                classification.
            default_classifications: Optional list of strings specifying default classification categories.
            merge_action: Optional OnExistingClassifications defining the action to be taken
                on existing classifications.
        """
        super().__init__(
            classification_store=classification_store,
            classification_label=classification_label,
            classification_scope=classification_scope or DEFAULT_SCOPE,
            num_samples=num_samples or DEFAULT_NUM_SAMPLES,
            num_iterations=num_iterations or DEFAULT_NUM_ITERATIONS,
            num_classifications=num_classifications or DEFAULT_NUM_CLASSIFICATIONS,
            splitter=splitter,
            llm=llm if llm and isinstance(llm, LLMCache) else LLMCache(
                llm=llm or GraphRAGConfig.extraction_llm,
                enable_cache=GraphRAGConfig.enable_cache
            ),
            prompt_template=prompt_template or DOMAIN_ENTITY_CLASSIFICATIONS_PROMPT,
            rank_prompt_template=prompt_template or RANK_ENTITY_CLASSIFICATIONS_PROMPT,
            default_classifications=default_classifications or DEFAULT_ENTITY_CLASSIFICATIONS,
            merge_action=merge_action or OnExistingClassifications.RETAIN_EXISTING
        )

        logger.debug(f'Prompt template: {self.prompt_template}')

    def _parse_classifications(self, response_text:str) -> Optional[List[str]]:
        """
        Parses domain-specific classifications from the response text.

        This function extracts classifications enclosed within the
        <entity_classifications> tags in the given response text. It returns
        a list of the parsed classifications or None if no classifications
        are found.

        Args:
            response_text (str): The response text containing classifications
                within <entity_classifications> HTML/XML tags.

        Returns:
            Optional[List[str]]: A list of classifications if successfully
                parsed from the response text, otherwise None.
        """
        pattern = r'<entity_classifications>(.*?)</entity_classifications>'
        match = re.search(pattern, response_text, re.DOTALL)

        classifications = []

        if match:
            classifications.extend([
                line.strip() 
                for line in match.group(1).strip().split('\n') 
                if line.strip()
            ])
                
        if classifications:
            logger.info(f'Successfully parsed {len(classifications)} domain-specific classifications')
            return classifications
        else:
            logger.warning(f'Unable to parse classifications from response: {response_text}')
            return classifications
            
       
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Parses the provided nodes for domain-specific classifications through multiple
        iterations while considering existing classifications and handling results based
        on the specified merge action. This method can leverage a splitter for chunking
        the input nodes and uses a language model for inferring classifications.

        Args:
            nodes (Sequence[BaseNode]): A sequence of node objects to be analyzed.
            show_progress (bool): Whether to show progress logs during the operation.
                Defaults to False.
            **kwargs (Any): Additional keyword arguments for future extensibility.

        Returns:
            List[BaseNode]: A list of nodes passed as input after completing the
            classification process.
        """

        seed_scope = f'{self.classification_scope}-seed'

        current_values = self.classification_store.get_scoped_values(self.classification_label, self.classification_scope)

        if current_values and self.merge_action == OnExistingClassifications.RETAIN_EXISTING:
            logger.info(f'Domain-specific classifications already exist [label: {self.classification_label}, scope: {self.classification_scope}, classifications: {current_values}]')
            return nodes
        
        seed_values = self.classification_store.get_scoped_values(self.classification_label, seed_scope)

        if not seed_values:
            seed_values = current_values
            self.classification_store.save_scoped_values(self.classification_label, seed_scope, seed_values)

        logger.info(f'Seed values    [label: {self.classification_label}, scope: {seed_scope}, classifications: {seed_values}]')   
        logger.info(f'Current values [label: {self.classification_label}, scope: {self.classification_scope}, classifications: {current_values}]')
        
        existing_classifications_map = {k:None for k in (seed_values + current_values)}
        existing_classifications = list(existing_classifications_map.keys())

        logger.info(f'Existing classifications: {existing_classifications}')

        chunks = self.splitter(nodes) if self.splitter else nodes

        classifications = set()

        for i in range(1, self.num_iterations + 1):

            sample_chunks = random.sample(chunks, self.num_samples) if len(chunks) > self.num_samples else chunks

            logger.info(f'Analyzing {len(sample_chunks)} chunks for domain adaptation [iteration: {i}, merge_action: {self.merge_action}]')

            formatted_chunks = '\n'.join(f'<chunk>{chunk.text}</chunk>' for chunk in sample_chunks)
                
            response = self.llm.predict(
                PromptTemplate(self.prompt_template),
                text_chunks=formatted_chunks,
                existing_classifications='\n'.join(existing_classifications) if OnExistingClassifications.MERGE_EXISTING else ''
            )

            classifications.update(self._parse_classifications(response))

        #if current_values and self.merge_action == OnExistingClassifications.MERGE_EXISTING:
        #    classifications.update(current_values)
            
        all_classifications = list(classifications)

        if all_classifications:
            
            formatted_classifications = '\n'.join([c.title() for c in all_classifications])
            response = self.llm.predict(
                PromptTemplate(self.rank_prompt_template),
                classifications=formatted_classifications
            )
            ranked_classifications = self._parse_classifications(response)[:self.num_classifications]

            logger.info(f'Domain adaptation succeeded [label: {self.classification_label}, scope: {self.classification_scope}, all_classifications: {all_classifications}, ranked_classification: {ranked_classifications}]')
            self.classification_store.save_scoped_values(self.classification_label, self.classification_scope, ranked_classifications)
        
        else:
            logger.warning(f'Domain adaptation failed, using default classifications [label: {self.classification_label}, scope: {self.classification_scope}, classifications: {self.default_classifications}]')
            self.classification_store.save_scoped_values(self.classification_label, self.classification_scope, self.default_classifications)

        return nodes
    
    def _parse_source_docs(self, source_documents):
        """
        Parses source documents and extracts their nodes for further parsing.

        This function takes a list of source documents, extracts their nodes,
        and passes them to an internal method `_parse_nodes` for further
        processing. The original list of source documents is returned after
        processing.

        Args:
            source_documents: List of source document objects. Each document should
                have a `nodes` attribute containing a list of nodes.

        Returns:
            list: The processed list of source document objects.
        """
        source_docs = [
            source_doc for source_doc in source_documents
        ]

        nodes = [
            n
            for sd in source_docs
            for n in sd.nodes
        ]

        self._parse_nodes(nodes)

        return source_docs