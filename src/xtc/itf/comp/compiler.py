from abc import ABC, abstractmethod
from ..schd.schedule import Schedule
from .module import Module
import xtc.itf

class Compiler(ABC):
    """An abstract implementation of a compiler for a given implementer and schedule.

    A Compiler takes a backend-specific implementation and schedule and generates
    an executable Module. It handles the final stage of converting the optimized
    intermediate representation into executable code for the target platform.
    """

    @abstractmethod
    def compile(self, schedule: Schedule) -> Module:
        """Compiles the implementation according to the given schedule.

        Args:
            schedule: The schedule specifying transformations and optimizations

        Returns:
            The compiled executable module
        """
        ...

    @property
    @abstractmethod
    def implementer(self) -> "xtc.itf.impl.Implementer":
        """Returns the implementer associated with this compiler.

        Returns:
            The implementer this compiler generates code for
        """
        ...
