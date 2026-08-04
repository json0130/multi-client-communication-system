"""
core/
=====
Application-agnostic platform code.

Nothing in this package may import from `robot/`, `gateway/`, `demo/` or `data/`,
and nothing here may contain scenario-specific logic. Subpackages:

  core.config    configuration singleton (`cfg`)
  core.rbac      role-based access control over the memory layer
  core.profiles  scenario profile loading + validation
"""
