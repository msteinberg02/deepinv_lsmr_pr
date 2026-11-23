from __future__ import annotations
from typing import Callable
import torch
from .utils import (
    _batched_dot as dot,
    _as_dim_list,
    _resolve_stagtol,
    _sample_shape,
    _make_scalar,
)
from deepinv.utils.tensorlist import zeros_like
from .conjugate_gradient_time import conjugate_gradient_time


def conjugate_gradient(
    A: Callable,
    b: torch.Tensor,
    A_true: Callable,
    b_true: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-6,
    stagtol: float | None = None,
    eps: float | None = None,
    parallel_dim: None | int | list[int] = 0,
    init: torch.Tensor | None = None,
    verbose: bool = False,
    x_true: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""
    Standard conjugate gradient algorithm.

    It solves the linear system :math:`Ax=b`, where :math:`A` is a (square) linear operator and :math:`b` is a tensor.

    For more details see: http://en.wikipedia.org/wiki/Conjugate_gradient_method

    .. note::
        A stagnation eps (configured via ``stagtol``) is used to prevent the algorithm from entering an infinite loop when numerical precision limits prevent further progress.
        This condition is verified via the norm of the search update: :math:`\frac{\|x_{k+1} - x_k\|_2}{\|x_k\|_2} \leq \text{stagtol}`.

    :param Callable A: Linear operator as a callable function, has to be square!
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param int max_iter: maximum number of CG iterations
    :param float tol: relative tolerance for stopping the CG algorithm based on the size of the residual: :math:`\frac{\|b - Ax_k\|_2}{\|b\|_2} \leq \text{tol}`.
    :param float stagtol: absolute tolerance for stopping the CG algorithm if iterates stagnate, default via dtype precision.
    :param float eps: a small value added to the (squared) denominators for numerical stability.
        If ``None`` (default), it is set precision-dependently to ``finfo(b.dtype).eps ** 2``,
        which guards against division by zero at convergence/breakdown without capping the
        attainable accuracy (a fixed constant here would floor the residual at ``~sqrt(eps)``).
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param torch.Tensor init: Optional initial guess.
    :param bool verbose: Output progress information in the console.
    :return: :class:`torch.Tensor` :math:`x` of shape (B, ...) verifying :math:`Ax=b`.
    """

    stagtol = _resolve_stagtol(stagtol, b)

    time = conjugate_gradient_time(
        A=A,
        b=b,
        max_iter=max_iter,
        tol=tol,
        stagtol=stagtol,
        eps=eps,
        parallel_dim=parallel_dim,
        init=init,
    )

    # Precision-dependent stabilization constant. It is added to squared denominators
    # (curvature ``<p, Ap>`` and ``||r||^2``), so using ``eps**2`` keeps the residual
    # floor at machine precision instead of a dtype-blind ``~sqrt(eps)``.
    if eps is None:
        eps = torch.finfo(b.dtype).eps ** 2

    parallel_dim = _as_dim_list(parallel_dim)
    shape = _sample_shape(b, parallel_dim)
    scalar = _make_scalar(shape, shape)

    if init is None:
        x = zeros_like(b)
    else:
        x = init.clone()

    r = b - A(x)
    p = r
    res_old = dot(r, r, parallel_dim).real
    b_norm_sq = dot(b, b, parallel_dim).real
    # handles case b=0
    b_norm_sq = torch.where(b_norm_sq > 0, b_norm_sq, torch.ones_like(b_norm_sq))
    stagtol = stagtol**2
    tol = b_norm_sq * (tol**2)

    alphas = []
    betas = []

    cond_list = []
    rlist = []
    error = []

    for i in range(int(max_iter)):
        Ap = A(p)
        alpha = res_old / (dot(p, Ap, parallel_dim) + eps)
        search_update = scalar(p, alpha, b_domain=True)
        x = x + search_update
        r = r - scalar(Ap, alpha, b_domain=True)
        res_new = dot(r, r, parallel_dim).real

        alphas.append(alpha.item())
        if i == 0:
            betas.append(0.0)  # Beta_0 is effectively 0
        else:
            betas.append((res_new / (res_old + eps)).item())

        k = i + 1
        T = torch.zeros((k, k), device="cpu")

        for j in range(k):
            if j == 0:
                T[j, j] = 1.0 / (alphas[j] + eps)
            else:
                T[j, j] = (1.0 / alphas[j] + eps) + (betas[j] / alphas[j - 1])
            if j > 0:
                off_diag = torch.sqrt(torch.tensor(betas[j])) / (alphas[j - 1] + eps)
                T[j, j - 1] = off_diag
                T[j - 1, j] = off_diag
        eigenvalues = torch.linalg.eigvalsh(T)
        max_eig = torch.max(eigenvalues).abs()
        min_eig = torch.min(eigenvalues).abs()

        cond_num = (max_eig / min_eig).item() if min_eig > eps else float("inf")
        cond_list.append(cond_num)

        search_update_norm = dot(search_update, search_update, parallel_dim).real
        xnorm = dot(x, x, parallel_dim).real

        resi = A_true(x) - b_true
        residual = dot(resi, resi, parallel_dim).real.sqrt()
        rlist.append(residual.detach().cpu().numpy().squeeze())
        x_err = x - x_true
        error.append(
            dot(x_err, x_err, parallel_dim).sqrt().detach().cpu().numpy().squeeze()
        )

        if torch.all(res_new <= tol):
            if verbose:
                print("CG converged at iteration", i + 1)
            break
        elif torch.all(search_update_norm <= stagtol * xnorm):
            if verbose:
                print("CG stagnated at iteration", i + 1)
            break
        p = r + scalar(p, res_new / (res_old + eps), b_domain=True)
        res_old = res_new

        # safeguard to avoid accumulating numerical errors in the computation of r
        # only applied every 100 iterations to limit computational overhead
        if i > 0 and i % 100 == 0:
            r = b - A(x)
            res_old = dot(r, r, parallel_dim).real
    else:
        if verbose:
            print("CG did not converge")

    return x, cond_list, rlist, error, time
