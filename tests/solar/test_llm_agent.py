# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression coverage for the agentic einsum converter's positive loop.

Paper SOLAR stage 2 (Agentic Einsum Converter): for an unseen operator the
agent generates a candidate handler, validates it by numerical emulation
against the real PyTorch operator, and only then adds it to the persistent
lookup table. These tests pin that generate -> verify -> cache -> reload
round-trip and the cache-integrity fail-closed branch, which the rest of the
suite exercises only through mocks.
"""

import hashlib
import json

import pytest

from solar.einsum.llm_agent import AgentConfig, NodeTypeConversionAgent
from solar.verification import VerificationError

# A handler that verify_generated_handler accepts for node_type "add" (mirrors
# the canonical fixture in test_verification_artifacts.py). Self-contained here
# to avoid cross-test-module imports.
VALID_ADD_HANDLER = """
def create_add_subgraph(node_id, node_data):
    return {
        node_id: {
            'type': 'add',
            'is_real_einsum': False,
            'tensor_names': {'inputs': ['left', 'right'], 'outputs': ['output']},
            'tensor_shapes': {
                'inputs': node_data['input_shapes'],
                'outputs': node_data['output_shapes'],
            },
            'tensor_dtypes': {
                'inputs': node_data['input_dtypes'],
                'outputs': node_data['output_dtypes'],
            },
        }
    }
"""

ADD_NODE_DATA = {
    "input_shapes": [[2], [2]],
    "output_shapes": [[2]],
    "input_dtypes": ["torch.float32", "torch.float32"],
    "output_dtypes": ["torch.float32"],
}


@pytest.fixture
def agent(tmp_path, monkeypatch):
    config = AgentConfig(api_key="sk-test", cache_dir=str(tmp_path / "cache"))
    instance = NodeTypeConversionAgent(config)
    # Avoid any real network call: the agent emulates the LLM returning a
    # known-good handler that the in-repo verifier then validates.
    monkeypatch.setattr(instance, "_call_llm", lambda prompt: VALID_ADD_HANDLER)
    return instance


class TestAgentGenerateVerifyCacheRoundTrip:
    def test_first_call_generates_verifies_and_caches(self, agent, tmp_path):
        code, metadata = agent.generate_conversion_code("add", ADD_NODE_DATA)

        assert metadata["source"] == "generated"
        assert metadata["verification"] == "passed"
        assert metadata["source_sha256"] == hashlib.sha256(code.encode()).hexdigest()

        cache_file = tmp_path / "cache" / "add.py"
        proof_file = tmp_path / "cache" / "add.verified.json"
        assert cache_file.exists() and proof_file.exists()
        assert cache_file.read_text() == code
        proof = json.loads(proof_file.read_text())
        assert proof["status"] == "passed"
        assert proof["source_sha256"] == hashlib.sha256(code.encode()).hexdigest()

    def test_second_call_serves_from_cache_and_revalidates(self, agent):
        first_code, first_meta = agent.generate_conversion_code("add", ADD_NODE_DATA)
        cached_code, cached_meta = agent.generate_conversion_code("add", ADD_NODE_DATA)

        assert first_meta["source"] == "generated"
        assert cached_meta["source"] == "cache"
        assert cached_meta["verification"] == "passed"
        # Cache hit must return byte-identical handler with a matching digest.
        assert cached_code == first_code
        assert cached_meta["source_sha256"] == first_meta["source_sha256"]

    def test_cache_tamper_triggers_fail_closed_revalidation(self, agent, monkeypatch):
        """A cached handler whose numerical revalidation now fails must be
        rejected, never silently served (cache-integrity fail-closed)."""
        agent.generate_conversion_code("add", ADD_NODE_DATA)  # populate cache

        import solar.verification as verification

        def _fail(_node_type, _code, _node_data):
            raise VerificationError("tampered handler no longer matches reference")

        monkeypatch.setattr(verification, "verify_generated_handler", _fail)

        with pytest.raises(RuntimeError, match="cached handler failed"):
            agent.generate_conversion_code("add", ADD_NODE_DATA)
