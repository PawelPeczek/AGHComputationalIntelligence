from __future__ import annotations

from abc import abstractmethod

from jmetal.util.termination_criterion import TerminationCriterion, \
    StoppingByEvaluations


class EnrichedTerminationCriterion(TerminationCriterion):

    @property
    @abstractmethod
    def current_value(self):
        pass

    @property
    @abstractmethod
    def progress(self):
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class EnrichedStoppingByEvaluations(StoppingByEvaluations, EnrichedTerminationCriterion):

    @property
    def value(self) -> int:
        return self.evaluations

    @property
    def target_value(self) -> int:
        return self.max_evaluations

    @property
    def progress(self) -> float:
        return self.evaluations / self.max_evaluations

    def reset(self) -> None:
        self.evaluations = 0

