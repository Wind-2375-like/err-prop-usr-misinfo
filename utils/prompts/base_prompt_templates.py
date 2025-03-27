from abc import ABC, abstractmethod

class BasePromptTemplates:
    def __init__(self):
        pass

    @abstractmethod
    def get_system_content_and_query(
        self,
        row_query,
        row_premise,
        row_perturbed_premise,
        column, 
        **kwargs
    ):
        """Abstract method to get the system content and query."""
        pass