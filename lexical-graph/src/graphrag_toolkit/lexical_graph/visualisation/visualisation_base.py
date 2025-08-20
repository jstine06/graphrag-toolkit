# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Optional

class VisualisationBase():
    @abc.abstractmethod
    def display_results(self, response, include_sources=True):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def display_schema(self, tenant_id:Optional[str]=None):
        raise NotImplementedError()