"""Runtime environment inspection for Events Diary.

This module provides a lightweight representation of the current
execution environment.

The runtime profile is intentionally limited to information available
from Python's standard library. Hardware-specific information such as
GPU availability, CUDA support, and VRAM will be added later when the
project introduces the relevant machine-learning dependencies.
"""

import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    """Basic information about the current execution environment.

    Attributes:
        python_version:
            Python interpreter version used to run the application.

        operating_system:
            Name of the operating system, such as Windows, Linux,
            or Darwin.

        machine:
            Machine architecture reported by the operating system,
            such as AMD64 or x86_64.

        processor:
            Processor information reported by the operating system.

    The profile is immutable because it represents the execution
    environment at the time it was collected.
    """

    python_version: str
    operating_system: str
    machine: str
    processor: str


def get_runtime_profile() -> RuntimeProfile:
    """Collect basic information about the current runtime environment.

    Returns:
        RuntimeProfile:
            A structured snapshot of the Python version, operating
            system, machine architecture, and processor information.

    This function relies only on Python's standard library so that
    runtime inspection remains lightweight during the early project
    phases.
    """

    return RuntimeProfile(
        python_version=sys.version.split()[0],
        operating_system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
    )