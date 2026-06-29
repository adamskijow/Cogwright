# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Cogwright: a local-first retrieval engine for technical equipment documentation.

The package is split into a pure ``core`` library that holds all retrieval and
decision logic behind protocol seams, and a thin ``adapters`` plus ``cli`` layer
that wires real input and output. The core has no dependency on any specific
model, vector database, or web framework.
"""

__version__ = "0.2.0"
