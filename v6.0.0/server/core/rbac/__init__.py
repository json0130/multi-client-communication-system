"""
core/rbac/
==========
Role-based access control over the multi-layered memory system.

Implements the RBAC design from Song et al., "Orchestrating Role-Based Social
Continuity for Heterogeneous Multi-Robot Teams":

  Global Access Level (Manager)  cross-client visibility over global records
                                 within its scenario.
  Local Access Level (Worker)    local isolation — its own records only, plus
                                 whatever a Manager explicitly delegates.

  Context Serialization          a Manager hands a Worker specific snippets for
                                 one task (grants.py) instead of widening the
                                 Worker's standing access.

Enforcement sits inside the memory layer, ahead of prompt assembly:

    stores -> RBACFilter -> Composite System Prompt

Layout:
  levels.py   AccessLevel / Visibility enums and fail-closed parsers
  models.py   RobotIdentity, MemoryRecord, Decision
  policy.py   pure decision functions, no I/O
  filter.py   the enforcement filter + the ClearedRecord clearance stamp
  grants.py   short-lived delegation grants
  audit.py    decision logging

This package is application-agnostic. It must not import from robot/, gateway/,
demo/ or data/, and must contain no scenario-specific logic.
"""

from core.rbac.audit import (
    AuditEvent,
    AuditSink,
    BatchingAuditSink,
    MemoryAuditSink,
    NullAuditSink,
    build_event,
)
from core.rbac.filter import (
    ClearanceError,
    ClearedRecord,
    RBACFilter,
    assert_cleared,
    make_record_id,
)
from core.rbac.grants import (
    DEFAULT_GRANT_TTL_SEC,
    DelegationGrant,
    GrantStore,
    new_grant,
)
from core.rbac.levels import (
    AccessLevel,
    InvalidAccessLevel,
    Visibility,
    parse_access_level,
    parse_visibility,
)
from core.rbac.models import Decision, MemoryRecord, RobotIdentity
from core.rbac.policy import Reason, can_read, default_visibility_for

__all__ = [
    "AccessLevel",
    "AuditEvent",
    "AuditSink",
    "BatchingAuditSink",
    "ClearanceError",
    "ClearedRecord",
    "DEFAULT_GRANT_TTL_SEC",
    "Decision",
    "DelegationGrant",
    "GrantStore",
    "InvalidAccessLevel",
    "MemoryAuditSink",
    "MemoryRecord",
    "NullAuditSink",
    "RBACFilter",
    "Reason",
    "RobotIdentity",
    "Visibility",
    "assert_cleared",
    "build_event",
    "can_read",
    "default_visibility_for",
    "make_record_id",
    "new_grant",
    "parse_access_level",
    "parse_visibility",
]
