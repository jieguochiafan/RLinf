# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared building blocks of the frozen ``v1`` rollout protocol."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar

SCHEMA_VERSION = "v1"
"""Wire-format version carried by every ``v1`` message.

Compatibility rules:

- A field set of a released version is frozen; ``tests/test_api_v1_schema.py``
  locks it so that accidental removals/renames fail CI.
- Purely additive changes (a new optional field with a default) stay inside
  ``v1`` only when older peers can ignore them safely.
- Anything else requires a new ``api/v2`` package; both versions may then be
  served side by side and distinguished by ``schema_version``.
"""


class SchemaError(ValueError):
    """Raised when a payload does not conform to the declared schema."""


@dataclass(kw_only=True)
class SchemaBase:
    """Base class of all ``v1`` wire types.

    Every subclass is a keyword-only dataclass carrying a ``schema_version``
    field so that a receiver can dispatch on the protocol version without
    out-of-band information.

    Subclasses that define ``__post_init__`` must call
    ``super().__post_init__()`` to keep version validation active.
    """

    SCHEMA_VERSION: ClassVar[str] = SCHEMA_VERSION

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.check_schema_version()

    def check_schema_version(self) -> None:
        """Validate the declared wire version.

        Raises:
            SchemaError: If ``schema_version`` is not the version implemented by
                this module.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"{type(self).__name__} declares schema_version="
                f"{self.schema_version!r}, but this module implements "
                f"{SCHEMA_VERSION!r}."
            )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the declared field names, in declaration order."""
        return tuple(f.name for f in dataclasses.fields(cls))

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow ``dict`` view of the message.

        Values are returned as-is (no deep copy), so tensors are not cloned.
        """
        return {name: getattr(self, name) for name in self.field_names()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaBase":
        """Rebuild a message from a shallow ``dict``.

        Args:
            data: Mapping produced by :meth:`to_dict` (or an equivalent decoder).

        Returns:
            The reconstructed message.

        Raises:
            SchemaError: If ``data`` contains keys that are not part of this
                version's field set.
        """
        unknown = set(data) - set(cls.field_names())
        if unknown:
            raise SchemaError(
                f"Unknown field(s) for {cls.__name__} ({SCHEMA_VERSION}): "
                f"{sorted(unknown)}"
            )
        return cls(**data)


@dataclass(kw_only=True)
class Metadata(SchemaBase):
    """Free-form, non-load-bearing annotations attached to a message.

    Anything a producer wants to pass through without becoming part of the
    frozen contract goes here. Consumers must treat unknown entries as opaque.

    Args:
        producer: Identity of the component that created the message, e.g.
            ``"rlinf_rollout.workers.rollout.hf"``.
        created_at: Unix timestamp (seconds) of message creation.
        tags: Arbitrary string annotations.
        extra: Arbitrary key/value annotations.
    """

    producer: str = ""
    created_at: float = 0.0
    tags: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
