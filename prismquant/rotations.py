"""Calibrated anchor rotations, normalized Hadamard transforms and compact WY.

Extracted from the validated research implementation; row-vector conventions
and serialized factor keys are preserved for released checkpoints.
"""
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable
import torch

_PALEY_CACHE = {}
PALEY_FP64_ORDERS = (68, 200)

def _fast_walsh_hadamard(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError(f"FWHT dimension must be a power of two, got {n}")
    original_shape = x.shape
    y = x.reshape(-1, n)
    width = 1
    while width < n:
        blocks = y.reshape(-1, n // (2 * width), 2, width)
        left = blocks[:, :, 0, :]
        right = blocks[:, :, 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(-1, n)
        width *= 2
    return (y / math.sqrt(n)).reshape(original_shape)


def _paley_hadamard_12(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q = 11
    residues = {value * value % q for value in range(1, q)}
    core = torch.empty((q, q), dtype=dtype, device=device)
    for row in range(q):
        for column in range(q):
            delta = (row - column) % q
            core[row, column] = 0 if delta == 0 else (1 if delta in residues else -1)
    matrix = torch.ones((q + 1, q + 1), dtype=dtype, device=device)
    matrix[1:, 1:] = core - torch.eye(q, dtype=dtype, device=device)
    error = (matrix @ matrix.T - (q + 1) * torch.eye(q + 1, dtype=dtype, device=device)).abs().max()
    if float(error) > 1e-5:
        raise AssertionError(f"invalid Paley-12 Hadamard: {float(error)}")
    return matrix / math.sqrt(q + 1)


def _full_hadamard_rows(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    signed = x * signs
    if not n & (n - 1):
        return _fast_walsh_hadamard(signed)
    if n == 3072:
        factored = _fast_walsh_hadamard(signed.reshape(-1, 12, 256))
        h12 = _paley_hadamard_12(x.device, x.dtype)
        return (factored.transpose(1, 2) @ h12.T).transpose(1, 2).reshape_as(x)
    raise ValueError(f"no frozen full-Hadamard construction for n={n}")


def _householders_to_anchors(vectors: torch.Tensor, b: int) -> tuple[list[torch.Tensor | None], float]:
    """Return sequential reflectors mapping column i to coordinate i*b."""
    work = vectors.float().clone()
    n, rank = work.shape
    reflectors: list[torch.Tensor | None] = []
    for index in range(rank):
        anchor = index * b
        target = torch.zeros(n, dtype=work.dtype, device=work.device)
        target[anchor] = 1.0
        delta = work[:, index] - target
        norm = delta.norm()
        if float(norm) < 1e-7:
            reflector = None
        else:
            reflector = delta / norm
            work[:, index:] -= 2 * reflector[:, None] * (reflector @ work[:, index:])[None, :]
        reflectors.append(reflector)
    anchors = torch.arange(rank, device=work.device) * b
    mapped = work[anchors, torch.arange(rank, device=work.device)]
    off = work.clone()
    off[anchors, torch.arange(rank, device=work.device)] = 0
    mapping_error = max(float((mapped - 1).abs().max()), float(off.abs().max()))
    return reflectors, mapping_error
def _balanced_target_slots(energies: list[float], groups: int, b: int) -> list[int]:
    loads = [0.0] * groups
    used = [0] * groups
    slots: list[int] = []
    for energy in energies:
        candidates = [g for g in range(groups) if used[g] < b - 1]
        group = min(candidates, key=lambda g: (loads[g], used[g], g))
        slot = group * b + 1 + used[group]
        used[group] += 1
        loads[group] += float(energy)
        slots.append(slot)
    return slots
def _gf27_mul(a: int, b: int) -> int:
    """Multiply in GF(3^3), modulus x^3 + 2x + 1."""
    ac = [(a // (3**i)) % 3 for i in range(3)]
    bc = [(b // (3**i)) % 3 for i in range(3)]
    prod = [0] * 5
    for i in range(3):
        for j in range(3):
            prod[i + j] = (prod[i + j] + ac[i] * bc[j]) % 3
    # x^3 = x + 2 and x^4 = x^2 + 2x over this modulus.
    for degree in (4, 3):
        coefficient = prod[degree] % 3
        if not coefficient:
            continue
        prod[degree] = 0
        prod[degree - 3] = (prod[degree - 3] + 2 * coefficient) % 3
        prod[degree - 2] = (prod[degree - 2] + coefficient) % 3
    return prod[0] + 3 * prod[1] + 9 * prod[2]


def _gf27_sub(a: int, b: int) -> int:
    coefficients = [((a // (3**i)) - (b // (3**i))) % 3 for i in range(3)]
    return coefficients[0] + 3 * coefficients[1] + 9 * coefficients[2]


def paley_hadamard_28(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    residues = {_gf27_mul(value, value) for value in range(1, 27)}
    if len(residues) != 13:
        raise AssertionError("GF(27) quadratic-residue construction failed")
    core = torch.empty((27, 27), device=device, dtype=dtype)
    for row in range(27):
        for column in range(27):
            delta = _gf27_sub(row, column)
            core[row, column] = 0 if delta == 0 else (1 if delta in residues else -1)
    matrix = torch.ones((28, 28), device=device, dtype=dtype)
    matrix[1:, 1:] = core - torch.eye(27, device=device, dtype=dtype)
    error = (matrix @ matrix.T - 28 * torch.eye(28, device=device, dtype=dtype)).abs().max()
    if float(error) > 1e-4:
        raise AssertionError(f"invalid Paley-28 Hadamard: {float(error)}")
    return matrix / math.sqrt(28)


def _is_prime(q: int) -> bool:
    return q > 1 and all(q % p for p in range(2, int(q ** 0.5) + 1))


def paley_hadamard(order: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A normalized Hadamard matrix of the given order by Paley's constructions.

    Paley I gives order q+1 for a prime q = 3 (mod 4); Paley II gives order
    2(q+1) for a prime q = 1 (mod 4). The Qwen3 family needs 20 (hidden 2560
    and 5120 are 20 x 2^k, and the 14B has 40 heads), 68 (17408 = 68 x 256) and
    76 (9728 = 76 x 128) and 200 (the 32B's 25600 = 200 x 128, Paley I with q = 199); the 12 and 28 the Llama models need keep their own
    constructions. Every matrix is verified as H H^T = n I before use.
    """
    key = (order, str(device), dtype)
    if key in _PALEY_CACHE:
        return _PALEY_CACHE[key]
    q = order - 1
    if _is_prime(q) and q % 4 == 3:
        residues = {value * value % q for value in range(1, q)}
        core = torch.empty((q, q), dtype=torch.float64)
        for row in range(q):
            for column in range(q):
                delta = (row - column) % q
                core[row, column] = 0 if delta == 0 else (1 if delta in residues else -1)
        matrix = torch.ones((order, order), dtype=torch.float64)
        matrix[1:, 1:] = core - torch.eye(q, dtype=torch.float64)
    elif order % 2 == 0 and _is_prime(order // 2 - 1) and (order // 2 - 1) % 4 == 1:
        q = order // 2 - 1
        residues = {value * value % q for value in range(1, q)}
        core = torch.empty((q, q), dtype=torch.float64)
        for row in range(q):
            for column in range(q):
                delta = (row - column) % q
                core[row, column] = 0 if delta == 0 else (1 if delta in residues else -1)
        conference = torch.zeros((q + 1, q + 1), dtype=torch.float64)
        conference[0, 1:] = 1
        conference[1:, 0] = 1
        conference[1:, 1:] = core
        eye = torch.eye(q + 1, dtype=torch.float64)
        matrix = torch.cat((torch.cat((conference + eye, conference - eye), 1),
                            torch.cat((conference - eye, -conference - eye), 1)), 0)
    else:
        raise ValueError(f"no Paley construction for Hadamard order {order}")
    error = (matrix @ matrix.T - order * torch.eye(order, dtype=torch.float64)).abs().max()
    if float(error) > 1e-9:
        raise AssertionError(f"invalid Paley Hadamard of order {order}: {float(error)}")
    result = (matrix / math.sqrt(order)).to(device=device, dtype=dtype)
    _PALEY_CACHE[key] = result
    return result


def full_hadamard_rows(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    quotient, remainder = divmod(n, 28)
    if not remainder and quotient >= 1 and not quotient & (quotient - 1):
        signed = x * signs
        factored = _fast_walsh_hadamard(signed.reshape(-1, 28, quotient))
        h28 = paley_hadamard_28(x.device, x.dtype)
        return (factored.transpose(1, 2) @ h28.T).transpose(1, 2).reshape_as(x)
    quotient, remainder = divmod(n, 12)
    if not remainder and quotient >= 1 and not quotient & (quotient - 1):
        signed = x * signs
        factored = _fast_walsh_hadamard(signed.reshape(-1, 12, quotient))
        h12 = _paley_hadamard_12(x.device, x.dtype)
        return (factored.transpose(1, 2) @ h12.T).transpose(1, 2).reshape_as(x)
    if n & (n - 1):
        # Qwen3 widths that are neither a power of two nor 12 or 28 times one.
        for order in (20, 68, 76, 200):
            quotient, remainder = divmod(n, order)
            if not remainder and quotient >= 1 and not quotient & (quotient - 1):
                signed = x * signs
                factored = _fast_walsh_hadamard(signed.reshape(-1, order, quotient))
                # The dense Paley factor is accumulated in fp64 for the two
                # largest orders (the 14B's 17408 = 68 x 256 and the 32B's
                # 25600 = 200 x 128): in fp32 its round trip lands at 1.3e-6,
                # over the fold contract's 1e-6. Orders 20 and 76 stay fp32,
                # which every finished row used.
                dtype = torch.float64 if order in PALEY_FP64_ORDERS else x.dtype
                h = paley_hadamard(order, x.device, dtype)
                return (factored.transpose(1, 2).to(dtype) @ h.T).to(x.dtype).transpose(1, 2).reshape_as(x)
    return _full_hadamard_rows(x, signs)


def balanced_orders(x_after_g: torch.Tensor, rank: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = x_after_g.shape[-1]
    groups = n // b
    energies = x_after_g.square().mean(0).double().cpu()
    absorbed_sources = [index * b for index in range(rank)]
    remaining = [index for index in range(n) if index not in absorbed_sources]
    fillers = sorted(remaining, key=lambda index: (float(energies[index]), index))[: groups - rank]
    anchor_sources = absorbed_sources + fillers
    residual_sources = [index for index in remaining if index not in fillers]
    residual_sources.sort(key=lambda index: (-float(energies[index]), index))
    residual_energies = [max(0.0, float(energies[index])) for index in residual_sources]
    target = [group * b for group in range(groups)] + _balanced_target_slots(residual_energies, groups, b)
    source = anchor_sources + residual_sources
    if len(source) != n or len(set(source)) != n or len(set(target)) != n:
        raise AssertionError("invalid calibrated permutation")
    return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def apply_reflectors(x: torch.Tensor, reflectors: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    output = x.float()
    for index in range(reflectors.shape[0]):
        if bool(active[index]):
            vector = reflectors[index]
            output = output - 2 * (output @ vector).unsqueeze(-1) * vector
    return output


def reflectors_from_vectors(vectors: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    refs, error = _householders_to_anchors(vectors.float(), b)
    active = torch.tensor([item is not None for item in refs], dtype=torch.bool)
    stacked = torch.stack([
        item if item is not None else torch.zeros(vectors.shape[0], dtype=torch.float32, device=vectors.device)
        for item in refs
    ]).float()
    return stacked, active.to(vectors.device), error


@dataclass
class RotationFactor:
    n: int
    b: int
    reflectors: torch.Tensor
    active: torch.Tensor
    source_order: torch.Tensor
    target_order: torch.Tensor
    anchor_error: float

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "RotationFactor":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            n=int(payload["n"]), b=int(payload["b"]),
            reflectors=payload["reflectors"].float().to(device),
            active=payload["active"].bool().to(device),
            source_order=payload["source_order"].long().to(device),
            target_order=payload["target_order"].long().to(device),
            anchor_error=float(payload["anchor_error"]),
        )

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "n": self.n, "b": self.b,
            "reflectors": self.reflectors.float().cpu(),
            "active": self.active.bool().cpu(),
            "source_order": self.source_order.long().cpu(),
            "target_order": self.target_order.long().cpu(),
            "anchor_error": self.anchor_error,
        }
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def apply(self, x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        rows = x.float().reshape(-1, self.n)
        rows = apply_reflectors(rows, self.reflectors, self.active)
        permuted = torch.empty_like(rows)
        permuted[:, self.target_order] = rows[:, self.source_order]
        signed = (permuted * signs).reshape(-1, self.n // self.b, self.b)
        return _fast_walsh_hadamard(signed).reshape(original_shape)


    def transpose(self, value: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        """Apply the true inverse: Walsh, signs, inverse permutation, reverse G."""
        shape = value.shape
        rows = _fast_walsh_hadamard(value.float().reshape(-1, self.n // self.b, self.b)).reshape(-1, self.n)
        rows = rows * signs
        restored = torch.empty_like(rows)
        restored[:, self.source_order] = rows[:, self.target_order]
        return apply_reflectors(restored, self.reflectors.flip(0), self.active.flip(0)).reshape(shape)


def factor_from_vectors(vectors: torch.Tensor, calibration_rows: torch.Tensor, b: int) -> RotationFactor:
    reflectors, active, error = reflectors_from_vectors(vectors, b)
    after_g = apply_reflectors(calibration_rows.float(), reflectors, active)
    source, target = balanced_orders(after_g, vectors.shape[1], b)
    return RotationFactor(
        n=vectors.shape[0], b=b, reflectors=reflectors, active=active,
        source_order=source.to(vectors.device), target_order=target.to(vectors.device),
        anchor_error=error,
    )
def compact_wy(reflectors: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return W,Y such that prod_i(I-2v_i v_i^T) = I-WY^T."""
    columns_w: list[torch.Tensor] = []
    columns_y: list[torch.Tensor] = []
    for index in range(reflectors.shape[0]):
        if not bool(active[index]):
            continue
        vector = reflectors[index].float()
        if columns_w:
            w = torch.stack(columns_w, dim=1)
            y = torch.stack(columns_y, dim=1)
            new_w = 2.0 * (vector - w @ (y.T @ vector))
        else:
            new_w = 2.0 * vector
        columns_w.append(new_w)
        columns_y.append(vector)
    if not columns_w:
        n = reflectors.shape[1]
        empty = reflectors.new_empty((n, 0), dtype=torch.float32)
        return empty, empty
    return torch.stack(columns_w, dim=1), torch.stack(columns_y, dim=1)


@dataclass
class WYFactor:
    factor: RotationFactor
    w: torch.Tensor
    y: torch.Tensor

    def apply_g(self, value: torch.Tensor) -> torch.Tensor:
        rows = value.float().reshape(-1, self.factor.n)
        projected = rows @ self.w
        return torch.addmm(rows, projected, self.y.T, beta=1.0, alpha=-1.0).reshape_as(value)

    def apply(self, value: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        rows = self.apply_g(value).reshape(-1, self.factor.n)
        permuted = torch.empty_like(rows)
        permuted[:, self.factor.target_order] = rows[:, self.factor.source_order]
        blocks = (permuted * signs).reshape(-1, self.factor.n // self.factor.b, self.factor.b)
        return _fast_walsh_hadamard(blocks).reshape(shape)
