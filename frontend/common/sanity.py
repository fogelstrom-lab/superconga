#!/usr/bin/env python3
# Copyright (C) 2019 and after, SuperConga team.
# All rights reserved.
# This file is part of the SuperConga Project, under the GNU LGPL v3
# license or higher. See LICENSE.txt for license information.

"""Functionality for doing sanity checks on parameters."""


from typing import Union

# Validate command-line arguments based on check list.
def check_value(key: str, value: Union[bool, float, int, str], checks: dict) -> bool:
    """Sanity check a parameter value."""

    # Check that the value is of the expected type. Or if the expected type is float, accept int as well.
    expected_type = checks["type"]
    is_float = expected_type == float
    is_expected = isinstance(value, expected_type) or (
        is_float and isinstance(value, int)
    )
    if not is_expected:
        print(f"-- ERROR! The field '{key}' must be of type {expected_type}.")
        return False

    # Run recursively if list or dictionary.
    if isinstance(value, list):
        success = True
        for elem in value:
            success &= check_value(key=key, value=elem, checks=checks["sub"])
        return success
    elif isinstance(value, dict):
        success = True
        success &= check_dict(config=value, checks=checks["fields"])
        return success

    # Check that the value is larger than 'low'.
    if "low" in checks:
        low = checks["low"]
        if not value > low:
            print(f"-- ERROR! The field '{key}' must be larger than {low}.")
            return False

    # Check that the value is larger than, or equal to, 'low'.
    if "low_include" in checks:
        low = checks["low_include"]
        if not value >= low:
            print(
                f"-- ERROR! The field '{key}' must be larger than, or equal to, {low}."
            )
            return False

    # Check that the value is smaller than 'high'.
    if "high" in checks:
        high = checks["high"]
        if not value < high:
            print(f"-- ERROR! The field '{key}' must be smaller than {high}.")
            return False

    # Check that the value is smaller than, or equal to, 'high'.
    if "high_include" in checks:
        high = checks["high_include"]
        if not value <= high:
            print(
                f"-- ERROR! The field '{key}' must be smaller than, or equal to, {high}."
            )
            return False

    # Check that the value is not in the excluded list.
    if "excluded" in checks:
        for exc in checks["excluded"]:
            if value == exc:
                print(f"-- ERROR! The field '{key}' cannot have value {exc}.")
                return False

    # Check that the value is in some pre-defined set.
    if "one_of" in checks:
        valid_values = checks["one_of"]
        if not value in valid_values:
            print(f"-- ERROR! The field '{key}' must be one of {valid_values}.")
            return False

    # No errors.
    return True


# Check that the necessary fields are set.
def check_dict(config: dict, checks: dict) -> bool:
    """Sanity check a dict."""

    success = True

    # Loop over the keys in the checks so we can spot if something is missing.
    for key, value_checks in checks.items():
        if not key in config:
            # Skip if the parameter is optional.
            if value_checks.get("optional", False):
                continue
            else:
                print(f"-- ERROR! The field '{key}' must exist.")
                success = False
                continue

        value = config[key]

        # Check if field is a dict: if so call the function again (recursively).
        if isinstance(value, dict):
            success &= check_dict(config=value, checks=value_checks["fields"])
        else:
            # If not a dict then check the values.
            success &= check_value(key=key, value=value, checks=value_checks)

    return success
