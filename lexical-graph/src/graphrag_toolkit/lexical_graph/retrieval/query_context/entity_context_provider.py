# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import time
import statistics
from typing import List, Dict

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.retrieval.model import ScoredEntity, EntityContexts, EntityContext
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorArgs

from llama_index.core.schema import QueryBundle


logger = logging.getLogger(__name__)

class EntityContextProvider():
    
    def __init__(self, graph_store:GraphStore, args:ProcessorArgs):
        self.graph_store = graph_store
        self.args = args
        
    def _get_entity_id_context_tree(self, entities:List[ScoredEntity]) -> Dict[str, Dict]:
        
        start = time.time()

        max_num_neighbours = self.args.ec_max_depth + 2
        
        entity_ids = [entity.entity.entityId for entity in entities if entity.score > 0] 
        
        excluded_entity_ids = set()
        entity_id_context_tree = {}
       
        for entity_id in entity_ids:

            if entity_id in excluded_entity_ids:
                continue
            else:
                excluded_entity_ids.add(entity_id)

            entity_id_context = {}
            entity_id_context_tree[entity_id] = entity_id_context
            
            start_entity_ids = set([entity_id])
            
            current_entity_id_contexts = { entity_id: entity_id_context  }

            for num_neighbours in range (max_num_neighbours, 2, -1):

                cypher = f"""
                // get next level in tree
                MATCH (entity:`__Entity__`)-[:`__RELATION__`]->(other)
                      -[r:`__SUBJECT__`|`__OBJECT__`]->()
                WHERE  {self.graph_store.node_id('entity.entityId')} IN $entityIds
                AND NOT {self.graph_store.node_id('other.entityId')} IN $excludeEntityIds
                WITH entity, collect(DISTINCT {self.graph_store.node_id('other.entityId')})[0..$numNeighbours] AS others, count(r) AS score ORDER BY score DESC
                RETURN {{
                    {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
                    others: others
                }} AS result    
                """

                params = {
                    'entityIds': list(start_entity_ids),
                    'excludeEntityIds': list(excluded_entity_ids),
                    'numNeighbours': num_neighbours
                }

                results = self.graph_store.execute_query(cypher, params)

                new_entity_id_contexts = {}

                for result in results:
                    
                    start_entity_id = result['result']['entity']['entityId']
                    other_entity_ids = result['result']['others']

                    for other_entity_id in other_entity_ids:
                        if other_entity_id in excluded_entity_ids:
                            continue
                        else:
                            excluded_entity_ids.add(other_entity_id)
                        child_context = { }
                        current_entity_id_contexts[start_entity_id][other_entity_id] = child_context
                        new_entity_id_contexts[other_entity_id] = child_context


                other_entity_ids = set([
                    other_id
                    for result in results
                    for other_id in result['result']['others'] 
                ])

                start_entity_ids = other_entity_ids

                current_entity_id_contexts = new_entity_id_contexts

        end = time.time()
        duration_ms = (end-start) * 1000

        logger.debug(f'entity_id_context_tree: {entity_id_context_tree} ({duration_ms:.2f} ms)')
                
        return entity_id_context_tree
    
    def _get_neighbour_entities(self, entity_id_context_tree:Dict[str, Dict]) -> List[ScoredEntity]:

        start = time.time()

        neighbour_entity_ids = set()

        def walk_tree(d):
            for entity_id, children in d.items():
                neighbour_entity_ids.add(entity_id)
                walk_tree(children)
            
        for _, d in entity_id_context_tree.items():
            walk_tree(d)
        
        logger.debug(f'neighbour_entity_ids: {list(neighbour_entity_ids)}')

        cypher = f"""
        // expand entities: score entities by number of relations
        MATCH (entity:`__Entity__`)-[r:`__SUBJECT__`|`__OBJECT__`]->()
        WHERE {self.graph_store.node_id('entity.entityId')} IN $entityIds
        WITH entity, count(r) AS score
        RETURN {{
            {node_result('entity', self.graph_store.node_id('entity.entityId'), properties=['value', 'class'])},
            score: score
        }} AS result
        """

        params = {
            'entityIds': list(neighbour_entity_ids)
        }

        results = self.graph_store.execute_query(cypher, params)

        neighbour_entities = [
            ScoredEntity.model_validate(result['result']) for result in results
        ]

        end = time.time()
        duration_ms = (end-start) * 1000
        
        logger.debug(f'neighbour_entities: {neighbour_entities} ({duration_ms:.2f} ms)')

        return neighbour_entities

        
    def _get_entity_contexts(self, entities:List[ScoredEntity], entity_id_context_tree:Dict[str, Dict], query_bundle:QueryBundle) -> List[List[ScoredEntity]]:

        start = time.time()
       
        all_entities = {
            entity.entity.entityId:entity for entity in entities
        }

        all_contexts_map = {}

        def context_id(context):
            return ':'.join([se.entity.entityId for se in context])

        def walk_tree_ex(current_context, d):
            if not d:
                all_contexts_map[context_id(current_context)] = current_context
            
            for entity_id, children in d.items():
                context = [c for c in current_context]
                if entity_id in all_entities:
                    context.append(all_entities[entity_id])
                walk_tree_ex(context, children)
                

        walk_tree_ex([], entity_id_context_tree)

        logger.debug(f'all_contexts_map: {all_contexts_map}')

        partial_path_keys = []
        
        for key in all_contexts_map.keys():
            for other_key in all_contexts_map.keys():
                if key != other_key and other_key.startswith(key):
                    partial_path_keys.append(key)

        for key in partial_path_keys:
            all_contexts_map.pop(key, None)

        all_contexts = [context for _, context in all_contexts_map.items()]

        logger.debug(f'all_contexts: {all_contexts}')

        deduped_contexts = self.dedup_contexts(all_contexts)

        logger.debug(f'deduped_contexts: {deduped_contexts}')

        ordered_contexts = self.order_context_subtrees(deduped_contexts)

        logger.debug(f'ordered_contexts: {ordered_contexts}')

        contexts = ordered_contexts[:self.args.ec_max_contexts]

        end = time.time()
        duration_ms = (end-start) * 1000

        logger.debug(f'contexts: {contexts} ({duration_ms:.2f} ms)')

        return contexts
    
    def dedup_contexts(self, contexts:List[List[ScoredEntity]]) ->  List[List[ScoredEntity]]:

        context_map = {
            ','.join([e.entity.value.lower() for e in context]):context
            for context in contexts
        }

        context_keys = sorted(list(context_map.keys()), key=len)
        
        surviving_contexts = {}

        for idx, context_key in enumerate(context_keys):
            keep = True
            for other_context_key in context_keys[idx+1:]:
                if other_context_key.startswith(context_key):
                    keep = False
                    break
            if keep:
                surviving_contexts[context_key] = context_map[context_key]
                
        deduped_contexts = []

        for k in context_map.keys():
            context = surviving_contexts.pop(k, None)
            if context:
                deduped_contexts.append(context)

        return deduped_contexts
    

    def order_contexts(self, contexts:List[List[ScoredEntity]]) ->  List[List[ScoredEntity]]:

        def score_context(context:List[ScoredEntity]):
            score = statistics.mean([e.score for e in context])
            reranking_score = statistics.mean([e.reranking_score for e in context])
            return score/reranking_score if reranking_score > 0 else 0
        
        context_map = {
            ','.join([e.entity.value.lower() for e in context]):context
            for context in contexts
        }

        scored_context_map = {
            k:score_context(v)
            for k,v in context_map.items()
        }

        return [
            context_map[k]
            for k, _ in sorted(scored_context_map.items(), key=lambda item: item[1], reverse=True)
        ]
    
    def order_context_subtrees(self, contexts:List[List[ScoredEntity]]) ->  List[List[ScoredEntity]]:

        context_subtree_map = {}

        for context in contexts:
            root_entity_id = context[0].entity.entityId
            if root_entity_id not in context_subtree_map:
                context_subtree_map[root_entity_id] = []
            context_subtree_map[root_entity_id].append(context)

        for root_entity_id in context_subtree_map.keys():
             context_subtree_map[root_entity_id] = self.order_contexts(context_subtree_map[root_entity_id])

        return [
            context
            for contexts in context_subtree_map.values()
            for context in contexts
        ]
    
    def filter_entities(self, entities:List[ScoredEntity]) -> List[ScoredEntity]:

        baseline_score=entities[0].score

        upper_score_threshold = baseline_score * self.args.ec_max_score_factor
        lower_score_threshhold = baseline_score * self.args.ec_min_score_factor

        logger.debug(f'upper_score_threshold: {upper_score_threshold}, lower_score_threshhold: {lower_score_threshhold}')

        def filter_entity(entity:ScoredEntity):
            allow = entity.score <= upper_score_threshold and entity.score >= lower_score_threshhold
            if not allow:
                logger.debug(f'Discarding entity: {entity.model_dump_json(exclude_unset=True, exclude_none=True, warnings=False)}')
            return allow

        filtered_entities = [
            e 
            for e in entities 
            if filter_entity(e)
        ]

        filtered_entities.sort(key=lambda e:e.score, reverse=True)

        logger.debug(f'filtered_entities: {filtered_entities}')

        return filtered_entities
             
    def get_entity_contexts(self, entities:List[ScoredEntity], query_bundle:QueryBundle)  -> EntityContexts:

        start = time.time()

        if entities:
        
            entity_id_context_tree = self._get_entity_id_context_tree(entities)
            
            neighbour_entities = self._get_neighbour_entities(
                entity_id_context_tree=entity_id_context_tree,
            )

            entities.extend(neighbour_entities)     

            entities = self.filter_entities(entities)
        
            entity_contexts = self._get_entity_contexts(
                entities=entities,
                entity_id_context_tree=entity_id_context_tree,
                query_bundle=query_bundle
            )

        else:
            entity_contexts = []

        end = time.time()
        duration_ms = (end-start) * 1000

        logger.debug(f"""Retrieved {len(entity_contexts)} entity contexts for '{query_bundle.query_str}' ({duration_ms:.2f} ms): {[
            str([e.entity.value for e in context])
            for context in entity_contexts
        ]}""")

        return EntityContexts(contexts=[EntityContext(entities=entities) for entities in entity_contexts])
