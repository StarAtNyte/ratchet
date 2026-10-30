"""Prefix-reuse infrastructure for the inner student loop (design doc §6).

Public surface:
    Decision      -- one inner-loop decision and the milestone entries it needs
    ResumePlan    -- where to resume + which checkpoints are invalidated
    compute_resume -- given a milestone patch, compute the resume plan
"""

from forgekernel.resume.invalidation import (
    Decision, ReopenPlan, ResumePlan, compute_reopen, compute_resume,
)

__all__ = ["Decision", "ReopenPlan", "ResumePlan", "compute_reopen", "compute_resume"]
