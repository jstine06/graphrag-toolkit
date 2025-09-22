# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging
import time
import concurrent.futures

from os.path import join
from datetime import datetime
from itertools import repeat
from typing import List, Any, Generator, Optional, Dict

from graphrag_toolkit.lexical_graph.indexing import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.model import SourceDocument, SourceType, source_documents_from_source_types
from graphrag_toolkit.lexical_graph.indexing.constants import PROPOSITIONS_KEY, TOPICS_KEY
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY
from graphrag_toolkit.lexical_graph import GraphRAGConfig

from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)

class S3BasedDocs(NodeHandler):
    """
    Handles S3-based document storage and retrieval for a specific collection.

    This class manages the interaction with Amazon S3 to store, organize, retrieve,
    and process documents with metadata and relationships. It supports filtering of
    metadata, iterates over collections of documents stored in S3, and saves
    processed documents back to S3. The class also provides encryption options for
    data storage.

    Attributes:
        region (str): AWS region where the S3 bucket is located.
        bucket_name (str): Name of the S3 bucket used for storage.
        key_prefix (str): Prefix path used to organize objects in the S3 bucket.
        collection_id (str): Unique identifier for the collection in S3.
        s3_encryption_key_id (Optional[str]): AWS KMS key ID for server-side encryption.
        metadata_keys (Optional[List[str]]): List of metadata keys to retain during metadata filtering.
    """
    region:str
    bucket_name:str
    key_prefix:str
    collection_id:str
    s3_encryption_key_id:Optional[str]=None
    metadata_keys:Optional[List[str]]=None

    def __init__(self, 
                 region:str, 
                 bucket_name:str, 
                 key_prefix:str, 
                 collection_id:Optional[str]=None,
                 s3_encryption_key_id:Optional[str]=None, 
                 metadata_keys:Optional[List[str]]=None):
        """
        Initializes an instance of the class with parameters for region, bucket name,
        key prefix, collection ID, S3 encryption key ID, and metadata keys. The
        collection ID defaults to the current timestamp if not provided.

        Args:
            region: The AWS region where the S3 bucket is hosted.
            bucket_name: The name of the S3 bucket to be used.
            key_prefix: The prefix to be applied to the keys for the objects stored.
            collection_id: The identifier for the collection. If not provided, defaults
                to a timestamp in '%Y%m%d-%H%M%S' format.
            s3_encryption_key_id: The ID of the encryption key used by S3 for encrypting
                objects.
            metadata_keys: A list of metadata keys associated with the collection. If
                None, metadata will not include additional keys.
        """
        super().__init__(
            region=region,
            bucket_name=bucket_name,
            key_prefix=key_prefix,
            collection_id=collection_id or datetime.now().strftime('%Y%m%d-%H%M%S'),
            s3_encryption_key_id=s3_encryption_key_id,
            metadata_keys=metadata_keys
        )

    def docs(self):
        """
        This function serves as a placeholder example and returns the instance on
        which it is called.

        Returns:
            The instance on which this method is invoked.
        """
        return self
    
    def _filter_metadata(self, node:TextNode) -> TextNode:
        """
        Filters the metadata within a TextNode object and its associated relationships to retain only
        specific keys. Deletes metadata keys that are neither in the allowed set of keys
        (PROPOSITIONS_KEY, TOPICS_KEY, INDEX_KEY) nor in the user-specified metadata keys (metadata_keys).

        Args:
            node (TextNode): The TextNode whose metadata and relationships' metadata will be filtered.

        Returns:
            TextNode: The filtered TextNode with irrelevant metadata keys removed.
        """
        def filter(metadata:Dict):
            """
            Handles operations on a TextNode object by filtering its metadata based on
            specified criteria. Utilizes predefined constants and optional metadata keys
            to clean the metadata attached to the given TextNode.

            This class inherits from NodeHandler, specializing its behavior to interact
            with the S3-based document storage or related metadata.

            Attributes:
                metadata_keys (Optional[List[str]]): A list of metadata keys that are allowed
                    to remain in the metadata dictionary. If None, filtering is based only on
                    predefined constants.
            """
            keys_to_delete = []
            for key in metadata.keys():
                if key not in [PROPOSITIONS_KEY, TOPICS_KEY, INDEX_KEY]:
                    if self.metadata_keys is not None and key not in self.metadata_keys:
                        keys_to_delete.append(key)
            for key in keys_to_delete:
                del metadata[key]

        filter(node.metadata)

        for _, relationship_info in node.relationships.items():
            if relationship_info.metadata:
                filter(relationship_info.metadata)

        return node
    
    def _download_chunk(self, chunk_key, s3_client):
        with io.BytesIO() as io_stream:
            s3_client.download_fileobj(self.bucket_name, chunk_key, io_stream)        
            io_stream.seek(0)
            data = io_stream.read().decode('UTF-8')
            return self._filter_metadata(TextNode.from_json(data))


    def __iter__(self):
        """
        Iterates through objects stored in an S3 bucket, retrieves source documents, processes
        their content, and generates `SourceDocument` objects.

        The method retrieves a collection of source document paths, iterates over them, retrieves chunks
        of data for each source document, processes the chunks to filter metadata, and finally yields
        `SourceDocument` objects containing a list of filtered nodes.

        Attributes:
            bucket_name (str): Name of the S3 bucket to access.
            key_prefix (str): Prefix for S3 object keys to locate the relevant collection data.
            collection_id (str): Identifier for the specific collection inside the S3 bucket.
            _filter_metadata (Callable): Callable for filtering metadata from a TextNode.

        Raises:
            Exception: Raised by boto3 or other libraries in case of issues with AWS S3 access or
                       data processing.

        Yields:
            SourceDocument: An object containing processed nodes as part of the source document.
        """
        s3_client = GraphRAGConfig.s3

        collection_path = join(self.key_prefix,  self.collection_id, '')

        paginator = s3_client.get_paginator('list_objects_v2')
        source_doc_pages = paginator.paginate(Bucket=self.bucket_name, Prefix=collection_path, Delimiter='/')

        source_doc_prefixes = [ 
            source_doc_obj['Prefix'] 
            for source_doc_page in source_doc_pages 
            for source_doc_obj in source_doc_page['CommonPrefixes']
             
        ]

        start = time.time()
        logger.debug(f'Started getting source documents from S3 [bucket: {self.bucket_name}, num_prefixes: {len(source_doc_prefixes)}]')

        doc_count = 0

        for source_doc_prefix in source_doc_prefixes:
            
            chunk_pages = paginator.paginate(Bucket=self.bucket_name, Prefix=source_doc_prefix)
            
            chunk_keys = [
                chunk_obj['Key']
                for chunk_page in chunk_pages
                for chunk_obj in chunk_page['Contents'] 
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=GraphRAGConfig.extraction_num_threads_per_worker) as executor:
                nodes = list(executor.map(
                    self._download_chunk,
                    chunk_keys,
                    repeat(s3_client)
                ))

            logger.debug(f'Yielding source document [source: {source_doc_prefix}, num_nodes: {len(nodes)}]')

            doc_count += 1
           
            yield SourceDocument(nodes=nodes)

        end = time.time()
        logger.debug(f'Finished getting {doc_count} source documents from S3 [bucket: {self.bucket_name}, prefixes: {source_doc_prefixes}] ({end - start} seconds)')

    def __call__(self, nodes: List[SourceType], **kwargs: Any) -> List[SourceDocument]:
        """
        Processes a list of source nodes and applies filtering or transformation based on additional
        input parameters passed through kwargs. This method evaluates each node via an acceptance
        mechanism and returns a filtered or transformed list of source documents.

        Args:
            nodes: A list of source nodes that will be processed.
            **kwargs: Arbitrary keyword arguments supporting additional options or configurations.

        Returns:
            List[SourceDocument]: A list of processed source documents derived from the input nodes.
        """
        return [n for n in self.accept(source_documents_from_source_types(nodes), **kwargs)]
    
    def _upload_chunk(self, root_path:str, n:TextNode, s3_client):
        chunk_output_path = join(root_path, f'{n.node_id}.json')
                    
        logger.debug(f'Writing chunk to S3: [bucket: {self.bucket_name}, key: {chunk_output_path}]')

        if self.s3_encryption_key_id:
            s3_client.put_object(
                Bucket=self.bucket_name,
                Key=chunk_output_path,
                Body=(bytes(json.dumps(n.to_dict(), indent=4).encode('UTF-8'))),
                ContentType='application/json',
                ServerSideEncryption='aws:kms',
                SSEKMSKeyId=self.s3_encryption_key_id
            )
        else:
            s3_client.put_object(
                Bucket=self.bucket_name,
                Key=chunk_output_path,
                Body=(bytes(json.dumps(n.to_dict(), indent=4).encode('UTF-8'))),
                ContentType='application/json',
                ServerSideEncryption='AES256'
            )

    def accept(self, source_documents: List[SourceDocument], **kwargs: Any) -> Generator[SourceDocument, None, None]:
        """
        Processes and uploads a list of source documents to an S3 bucket.

        The function iterates through a list of source documents, writes their contents
        to an S3 bucket, and yields the processed documents. It ensures that document
        nodes without specific metadata are uploaded as JSON files. Additionally, it
        supports both KMS and AES256 server-side encryption for file storage, depending
        on the presence of a provided encryption key.

        Args:
            source_documents (List[SourceDocument]): A list of source documents to be processed and uploaded.
            **kwargs (Any): Additional keyword arguments that may be passed but are not used within the function.

        Yields:
            SourceDocument: The processed source documents after they are written to S3.

        """

        s3_client = GraphRAGConfig.s3
        collection_prefix = join(self.key_prefix, self.collection_id)

        start = time.time()
        logger.debug(f'Started writing source documents to S3 [bucket: {self.bucket_name}, prefix: {collection_prefix}]')

        doc_count = 0

        for source_document in source_documents:
            
            root_path =  join(collection_prefix, source_document.source_id())
            logger.debug(f'Writing source document to S3 [bucket: {self.bucket_name}, prefix: {root_path}]')

            with concurrent.futures.ThreadPoolExecutor(max_workers=GraphRAGConfig.extraction_num_threads_per_worker) as executor:
                for n in source_document.nodes:
                    if not [key for key in [INDEX_KEY] if key in n.metadata]:
                        executor.submit(self._upload_chunk, root_path, n, s3_client)

            doc_count += 1

            yield source_document

        end = time.time()
        logger.debug(f'Finished writing {doc_count} source documents to S3 [bucket: {self.bucket_name}, prefix: {collection_prefix}] ({end - start} seconds)')
