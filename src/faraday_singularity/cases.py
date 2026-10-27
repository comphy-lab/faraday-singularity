"""Physics and numerical admission checks for one Faraday case."""

import math


def case_problems(case: dict) -> list[str]:
    problems = []
    defaults = {"fluid": {"Oh": 0.04, "Bo": 0.0},
                "container": {"mode": 1, "depth": 2 * math.pi},
                "forcing": {"Gamma": 0.0, "detuning": 1.0, "phase": 0.0},
                "initial": {"amplitude": -1e-3}}
    physics = case.get("physics", {})
    if not isinstance(physics, dict):
        return ["physics must be an object"]
    for block, values in defaults.items():
        given = physics.get(block, {})
        if not isinstance(given, dict):
            problems.append(f"physics.{block} must be an object")
            continue
        for key, default in values.items():
            value = given.get(key, default)
            field = f"physics.{block}.{key}"
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"{field} must be finite")
            elif key in {"Oh", "depth", "detuning"} and value <= 0:
                problems.append(f"{field} must be positive")
            elif key in {"Bo", "Gamma"} and value < 0:
                problems.append(f"{field} must be non-negative")
            elif key == "mode" and (not isinstance(value, int) or value < 1):
                problems.append(f"{field} must be a positive integer")
    for key in physics.keys() - defaults.keys():
        problems.append(f"physics.{key} is not used by the Faraday solver")
    return problems


def validate_case(case: dict) -> None:
    problems = case_problems(case)
    if problems:
        raise SystemExit("invalid Faraday case: " + "; ".join(problems))
