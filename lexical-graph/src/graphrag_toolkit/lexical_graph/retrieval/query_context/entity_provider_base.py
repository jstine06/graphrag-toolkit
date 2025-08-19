# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import time
import logging
from typing import List, Optional

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs

logger = logging.getLogger(__name__)

class EntityProviderBase():
    
    def __init__(self, graph_store:GraphStore, args:ProcessorArgs, filter_config:Optional[FilterConfig]=None):
        self.graph_store = graph_store
        self.args = args
        self.filter_config = filter_config

    @abc.abstractmethod                 
    def _get_entities(self, keywords:List[str])  -> List[ScoredEntity]:
        raise NotImplementedError

    def get_entities(self, keywords:List[str])  -> List[ScoredEntity]:
        
        start = time.time()
        entities = self._get_entities(keywords)
        end = time.time()
        duration_ms = (end-start) * 1000

        logger.debug(f"""[{type(self).__name__}] Entities ({duration_ms:.2f} ms): {[
            entity.model_dump_json(exclude_unset=True, exclude_defaults=True, exclude_none=True, warnings=False) 
            for entity in entities
        ]}""")

        return entities[:self.args.ec_num_entities]

        