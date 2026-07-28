# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Parsing and validation for extended-einsum equations."""


def parse_dim_tokens(dims_str: str, validate: bool = True) -> list[str]:
    """Parse uppercase rank tokens and parenthesized compound dimensions.

    Args:
        dims_str: Compact dimension notation such as ``A1BC(P+R)``.
        validate: Whether to reject repeated tokens.

    Returns:
        Individual uppercase dimension tokens.

    Raises:
        ValueError: If validation finds a repeated token.
    """
    if not dims_str:
        return []

    tokens = []
    i = 0
    while i < len(dims_str):
        # Handle parenthesized groups like (P+R)
        if dims_str[i] == "(":
            # Find matching closing parenthesis
            j = i + 1
            depth = 1
            while j < len(dims_str) and depth > 0:
                if dims_str[j] == "(":
                    depth += 1
                elif dims_str[j] == ")":
                    depth -= 1
                j += 1
            # Extract the group including parentheses, uppercase the content
            group = dims_str[i:j].upper()
            tokens.append(group)
            i = j
            continue

        if not dims_str[i].isalpha():
            # Skip non-alphabetic characters
            i += 1
            continue

        # Get the single letter (multi-letter prefixes NOT allowed)
        letter = dims_str[i].upper()
        i += 1

        # Check if followed by digits
        if i < len(dims_str) and dims_str[i].isdigit():
            # Collect all following digits
            j = i
            while j < len(dims_str) and dims_str[j].isdigit():
                j += 1
            digits = dims_str[i:j]
            tokens.append(letter + digits)
            i = j
        else:
            # No digits following - just the single letter
            tokens.append(letter)

    # Validate: no repeated ranks allowed in the same tensor
    # For parenthesized groups, we check the whole group as a token
    if validate and len(tokens) != len(set(tokens)):
        seen = set()
        duplicates = []
        for token in tokens:
            if token in seen:
                duplicates.append(token)
            seen.add(token)
        raise ValueError(
            f"Repeated rank(s) in tensor dimensions: {duplicates}. "
            f"Each dimension must be unique. Got: {tokens}",
        )

    return tokens


def validate_dim_tokens(
    tokens: list[str],
    raise_on_error: bool = False,
) -> bool:
    """Validate that dimension tokens have no duplicates (repeated ranks).

    Each dimension in a tensor must be unique. Repeated ranks like ["A", "A"]
    are semantically invalid.

    Args:
        tokens: List of dimension tokens to validate.
        raise_on_error: If True, raise ValueError on duplicates instead of returning False.

    Returns:
        True if all tokens are unique, False if there are duplicates.

    Raises:
        ValueError: If raise_on_error=True and there are duplicate tokens.

    Examples:
        validate_dim_tokens(["A", "B", "C"]) -> True
        validate_dim_tokens(["A", "A"]) -> False (repeated rank)
        validate_dim_tokens(["A0", "A1", "B0"]) -> True

    """
    if len(tokens) == len(set(tokens)):
        return True

    if raise_on_error:
        seen = set()
        duplicates = []
        for token in tokens:
            if token in seen:
                duplicates.append(token)
            seen.add(token)
        raise ValueError(
            f"Repeated rank(s) in tensor dimensions: {duplicates}. "
            f"Each dimension must be unique. Got: {tokens}",
        )

    return False


def parse_einsum_equation(
    equation: str,
) -> tuple[list[list[str]], list[str]]:
    """Parse an einsum equation into input operand tokens and output tokens.

    Tokens are in the format: single capital letter optionally followed by digit(s).
    Examples: A, B, A1, B1, A2, Z99, etc.

    Examples:
        "ABC,DE->ADE" -> ([["A", "B", "C"], ["D", "E"]], ["A", "D", "E"])
        "A1B1C1,D1E1->A1D1E1" -> ([["A1", "B1", "C1"], ["D1", "E1"]], ["A1", "D1", "E1"])
        "->ABC" -> ([], ["A", "B", "C"])  # start node
        "->A1B1C1" -> ([], ["A1", "B1", "C1"])  # start node with numbered dims

    Args:
        equation: Einsum equation string

    Returns:
        Tuple of (list of input operand token lists, output tokens)

    """
    if not equation or "->" not in equation:
        return [], []

    parts = equation.split("->")
    if len(parts) != 2:
        return [], []

    lhs, rhs = parts[0].strip(), parts[1].strip()

    # Parse output tokens
    output_tokens = parse_dim_tokens(rhs)

    # Parse input operands (comma-separated)
    input_operands: list[list[str]] = []
    if lhs:
        for raw_operand in lhs.split(","):
            operand_str = raw_operand.strip()
            if operand_str:
                input_operands.append(parse_dim_tokens(operand_str))

    return input_operands, output_tokens


def validate_einsum_ranks_match_shapes(
    equation: str,
    tensor_shapes: dict[str, list[list[int]]],
) -> tuple[bool, str]:
    """Validate that einsum equation ranks match tensor shapes.

    This function checks that the number of dimensions in each operand of the
    einsum equation matches the corresponding tensor shape.

    Args:
        equation: Einsum equation string (e.g., "AB,BC->AC")
        tensor_shapes: Dictionary with "inputs" and "outputs" keys, each containing
                      a list of shapes. Format: {"inputs": [[32, 64], [64, 128]], "outputs": [[32, 128]]}

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.

    Examples:
        >>> validate_einsum_ranks_match_shapes(
        ...     "AB,BC->AC",
        ...     {"inputs": [[32, 64], [64, 128]], "outputs": [[32, 128]]},
        ... )
        (True, "")
        >>> validate_einsum_ranks_match_shapes(
        ...     "AB,AB->AB", {"inputs": [[32, 64], [64]], "outputs": [[32, 64]]}
        ... )
        (False, "Einsum input operand 1 has 2 dims (AB) but tensor has shape [64] (1 dims)")

    """
    if not equation or "->" not in equation:
        return True, ""  # Can't validate without proper equation

    input_operands, output_tokens = parse_einsum_equation(equation)

    # Get input and output shapes from tensor_shapes
    input_shapes = tensor_shapes.get("inputs", [])
    output_shapes = tensor_shapes.get("outputs", [])

    errors = []

    # Validate input operands
    for i, operand_tokens in enumerate(input_operands):
        if i >= len(input_shapes):
            continue  # Skip if shape not available

        shape = input_shapes[i]
        if shape is None:
            continue

        expected_rank = len(operand_tokens)
        actual_rank = len(shape)

        if expected_rank != actual_rank:
            operand_str = "".join(operand_tokens)
            errors.append(
                f"Einsum input operand {i} has {expected_rank} dims ({operand_str}) "
                f"but tensor has shape {shape} ({actual_rank} dims)",
            )

    # Validate output operand
    if output_tokens and output_shapes:
        output_shape = output_shapes[0] if output_shapes else None
        if output_shape is not None:
            expected_rank = len(output_tokens)
            actual_rank = len(output_shape)

            if expected_rank != actual_rank:
                output_str = "".join(output_tokens)
                errors.append(
                    f"Einsum output has {expected_rank} dims ({output_str}) "
                    f"but tensor has shape {output_shape} ({actual_rank} dims)",
                )

    if errors:
        return False, "; ".join(errors)
    return True, ""


def validate_tensor_names_match_shapes(
    tensor_names: dict[str, list[str]],
    tensor_shapes: dict[str, list[list[int]]],
) -> tuple[bool, str]:
    """Validate that tensor_names and tensor_shapes have matching counts.

    This function checks that the number of tensor names matches the number of
    tensor shapes for both inputs and outputs.

    Args:
        tensor_names: Dictionary with "inputs" and "outputs" keys, each containing
                     a list of tensor names. Format: {"inputs": ["A", "B"], "outputs": ["C"]}
        tensor_shapes: Dictionary with "inputs" and "outputs" keys, each containing
                      a list of shapes. Format: {"inputs": [[32, 64], [64, 128]], "outputs": [[32, 128]]}

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.

    Examples:
        >>> validate_tensor_names_match_shapes(
        ...     {"inputs": ["A", "B"], "outputs": ["C"]},
        ...     {"inputs": [[32, 64], [64, 128]], "outputs": [[32, 128]]},
        ... )
        (True, "")
        >>> validate_tensor_names_match_shapes(
        ...     {"inputs": ["A", "B"], "outputs": ["C"]},
        ...     {"inputs": [[32, 64]], "outputs": [[32, 128]]},
        ... )
        (False, "Input tensor_names has 2 entries but tensor_shapes has 1")

    """
    errors = []

    # Validate inputs
    input_names = tensor_names.get("inputs", [])
    input_shapes = tensor_shapes.get("inputs", [])

    if len(input_names) != len(input_shapes):
        errors.append(
            f"Input tensor_names has {len(input_names)} entries "
            f"but tensor_shapes has {len(input_shapes)}",
        )

    # Validate outputs
    output_names = tensor_names.get("outputs", [])
    output_shapes = tensor_shapes.get("outputs", [])

    if len(output_names) != len(output_shapes):
        errors.append(
            f"Output tensor_names has {len(output_names)} entries "
            f"but tensor_shapes has {len(output_shapes)}",
        )

    if errors:
        return False, "; ".join(errors)
    return True, ""
