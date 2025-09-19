# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
import json
from typing import List, Dict, Any, Annotated, Optional, Union
from pydantic import Field

from graphrag_toolkit.lexical_graph import LexicalGraphQueryEngine
from graphrag_toolkit.lexical_graph import TenantId, TenantIdType, to_tenant_id, DEFAULT_TENANT_ID, DEFAULT_TENANT_NAME
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.retrieval.summary import GraphSummary, get_domain

logger = logging.getLogger(__name__)

def tool_search(graph_store:GraphStore, tenant_ids:List[TenantId]):

    def search_for_tool(search_term: Annotated[str, Field(description='Entity, concept or phrase for which one or more tools are to be found')]) -> List[str]:
        
        permitted_tools = [str(tenant_id) for tenant_id in tenant_ids]
        
        cypher = '''MATCH (n) 
        WHERE n.search_str STARTS WITH $search_term
        RETURN DISTINCT labels(n) AS lbls
        '''

        properties = {
            'search_term': search_term.lower()
        }

        results = graph_store.execute_query(cypher, properties)

        tool_names = set()

        for result in results:
            for label in result['lbls']:
                parts = label.split('__')
                if len(parts) == 4:
                    tool_names.add(parts[2])
                elif len(parts) == 3:
                    tool_names.add(str(DEFAULT_TENANT_ID))

        tools = [
            t 
            for t in list(tool_names) 
            if t in permitted_tools
        ]

        logger.debug(f'{search_term}: {tools}')

        return tools

    return search_for_tool

def query_tenant_graph(graph_store:GraphStore, vector_store:VectorStore, tenant_id:TenantId, domain:str, **kwargs):
    
    description = f'A natural language query related to the domain of {domain}' if domain else 'A natural language query'
    
    def query_graph(query: Annotated[str, Field(description=description)]) -> List[Dict[str, Any]]:
        
        query_engine = LexicalGraphQueryEngine.for_traversal_based_search(
            graph_store, 
            vector_store,
            tenant_id=tenant_id,
            enable_multipart_queries=True,
            **kwargs
        )

        start = time.time()

        response = query_engine.retrieve(query)
        
        end = time.time()

        results = [json.loads(n.text) for n in response]

        logger.debug(f'[{tenant_id}]: {query} [{len(results)} results, {int((end-start) * 1000)} millis]')
        
        return results
        
    return query_graph

def get_tenant_ids(graph_store:GraphStore):
    
    cypher = '''MATCH (n)
    WITH DISTINCT labels(n) as lbls
    WITH split(lbls[0], '__') AS lbl_parts WHERE size(lbl_parts) > 2
    WITH lbl_parts WHERE lbl_parts[1] = 'SYS_Class' AND lbl_parts[2] <> ''
    RETURN DISTINCT lbl_parts[2] AS tenant_id
    '''

    results = graph_store.execute_query(cypher)

    return [result['tenant_id'] for result in results]

TenantConfigType = Union[List[TenantIdType], Dict[str, Dict[str, Any]]]

"""
Config: 

{
    '<tenant_id>': {
        'description': '<short description - optional>',
        'refresh': True|False,
        'args': {
            LexicalGraphQueryEngine args
        }
    }
}
"""

def create_mcp_server(graph_store:GraphStore, vector_store:VectorStore, tenant_ids:Optional[TenantConfigType]=None, **kwargs):

    try:
        from fastmcp import FastMCP
        from fastmcp.tools import Tool
    except ImportError as e:
        raise ImportError(
            "fastmcp package not found, install with 'pip install fastmcp'"
        ) from e

    mcp = FastMCP(name='LexicalGraphServer')

    graph_summary = GraphSummary(graph_store)

    tenant_id_configs = {}

    if not tenant_ids:
        tenant_id_configs[DEFAULT_TENANT_NAME] = {}
        tenant_id_configs.update({tenant_id:{} for tenant_id in get_tenant_ids(graph_store)})
    else:
        if isinstance(tenant_ids, list):
            tenant_id_configs.update({str(tenant_id):{} for tenant_id in tenant_ids})
        else:
            tenant_id_configs.update(tenant_ids)
    
    for tenant_id_name, tenant_id_config in tenant_id_configs.items():

        tenant_id = to_tenant_id(tenant_id_name)

        refresh = tenant_id_config.get('refresh', False)
        
        summary = graph_summary.create_graph_summary(tenant_id, tenant_id_config.get('description', ''), refresh=refresh)

        if summary:

            domain = get_domain(summary)

            args = kwargs.copy()
            args.update(tenant_id_config.get('args', {}))

            logger.debug(f'Adding tool: [tenant_id: {tenant_id}, domain: {domain}, args: {args}]')


            mcp.add_tool(
                Tool.from_function(
                    fn=query_tenant_graph(graph_store, vector_store, tenant_id, domain, **args),
                    name = str(tenant_id),
                    description = summary
                )
                
            )

    if tenant_ids:
        mcp.add_tool(
            Tool.from_function(
                fn=tool_search(graph_store, tenant_ids),
                name = 'search_',
                description = 'Given a search term, returns the name of one or more tools that can be used to provide information about the search term. Use this tool to help find other tools that can answer a query.'
            )
            
        )

    return mcp
