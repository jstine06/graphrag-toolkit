# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class OnExistingClassifications(Enum):
    """
    Enumeration to define behaviors for handling existing classifications.

    This class is an Enum used to specify how existing classifications
    should be handled during operations. It provides three options:
    merging with existing classifications, replacing them entirely,
    or retaining the existing classifications without changes.

    Attributes:
        MERGE_EXISTING: Merges new classifications with the existing ones.
        REPLACE_EXISTING: Replaces any existing classifications completely.
        RETAIN_EXISTING: Keeps the existing classifications without changes.
    """
    MERGE_EXISTING = 1
    REPLACE_EXISTING = 2
    RETAIN_EXISTING = 3

@dataclass
class InferClassificationsConfig:
    """
    Configuration for inferring classifications in a data processing context.

    This class encapsulates the configuration parameters required for inferring
    classifications within a system. It defines the number of samples to process,
    the number of iterations to perform, how to handle existing classifications,
    and an optional prompt template for customization.

    Attributes:
        num_samples (Optional[int]): Number of samples to infer classifications from.
            Defaults to 5.
        num_iterations (Optional[int]): Number of iterations to perform for the
            classification inference process. Defaults to 1.
        num_classifications (Optional[int]): Maximum number of classifications to return.
            Defaults to 15.
        on_existing_classifications (Optional[OnExistingClassifications]): Strategy
            to apply when handling pre-existing classifications. Defaults to
            OnExistingClassifications.MERGE_EXISTING.
        prompt_template (Optional[str]): Custom template text for classification
            prompts, if applicable. Defaults to None.
    """
    num_samples:Optional[int]=5
    num_iterations:Optional[int]=1
    num_classifications:Optional[int]=15
    on_existing_classifications:Optional[OnExistingClassifications]=OnExistingClassifications.MERGE_EXISTING
    prompt_template:Optional[str]=None