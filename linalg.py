from torch import tensor, stack, float64, float32, profiler, device, tensordot
from torch import linalg
import torch
import numpy as np
from hamiltonian import Sky_phi
from hamiltonian import ham_total
from xitorch import linalg
from test_sparse_eigen import CsrLinOp

#see https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html
def cond(pred, true_fun, false_fun, *operands):
    if pred:
      return true_fun(*operands)
    else:
      return false_fun(*operands)

#see https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html?highlight=while_loop#jax.lax.while_loop
def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
      val = body_fun(val)
    return val

#see https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html?highlight=fori_loop
def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val

def shifted_QR(Vm, Hm, fm, shifts, numeig, device = "cuda"):
    # compress arnoldi factorization
    q = torch.tensor((), dtype=Hm.dtype, device = device).new_zeros((Hm.size(0)))
    q[-1] = 1.0

    def body(i, vals):
      Vm, Hm, q = vals
      shift = shifts[i] * torch.eye(Hm.size()[0], dtype=Hm.dtype, device = device)
      Qj, R = torch.linalg.qr(Hm - shift)
      Hm = R @ Qj + shift
      Vm = Qj.T @ Vm
      q = q @ Qj
      return Vm, Hm, q

    Vm, Hm, q = fori_loop(0, shifts.shape[0], body,
                                  (Vm, Hm, q))
    fk = Vm[numeig, :] * Hm[numeig, numeig - 1] + fm * q[numeig - 1]
    return Vm, Hm, fk

def iterative_classical_gram_schmidt(vector, krylov_vectors, precision, iterations = 2):
    """
    Orthogonalize `vector`  to all rows of `krylov_vectors`.
    Args:
      vector: Initial vector.
      krylov_vectors: Matrix of krylov vectors, each row is treated as a
        vector.
      iterations: Number of iterations.
    Returns:
      jax.ShapedArray: The orthogonalized vector.
    """
    i1 = list(range(1, len(krylov_vectors.shape)))
    i2 = list(range(len(vector.shape)))

    vec = vector
    overlaps = 0

    for _ in range(iterations):
      ov = tensordot(krylov_vectors.conj(), vec, (i1, i2))
      vec = vec - tensordot(ov, krylov_vectors, ([0], [0]))
      overlaps = overlaps + ov

    return vec, overlaps

def _lanczos_fact(matvec, args, v0, Vm, alphas, betas, start, num_krylov_vecs, tol, precision, device = "cuda"):
    """
    Compute an m-step lanczos factorization of `matvec`, with
    m <=`num_krylov_vecs`. The factorization will
    do at most `num_krylov_vecs` steps, and terminate early 
    if an invariat subspace is encountered. The returned arrays
    `alphas`, `betas` and `Vm` will satisfy the Lanczos recurrence relation
    ```
    matrix @ Vm - Vm @ Hm - fm * em = 0
    ```
    with `matrix` the matrix representation of `matvec`, 
    `Hm = jnp.diag(alphas) + jnp.diag(betas, -1) + jnp.diag(betas.conj(), 1)`
    `fm=residual * norm`, and `em` a cartesian basis vector of shape 
    `(1, kv.shape[1])` with `em[0, -1] == 1` and 0 elsewhere.
    Note that the caller is responsible for dtype consistency between
    the inputs, i.e. dtypes between all input arrays have to match.
    Args:
      matvec: The matrix vector product.
      args: List of arguments to `matvec`.
      v0: Initial state to `matvec`.
      Vm: An array for storing the krylov vectors. The individual
        vectors are stored as columns.
        The shape of `krylov_vecs` has to be
        (num_krylov_vecs + 1, np.ravel(v0).shape[0]).
      alphas: An array for storing the diagonal elements of the reduced
        operator.
      betas: An array for storing the lower diagonal elements of the 
        reduced operator.
      start: Integer denoting the start position where the first
        produced krylov_vector should be inserted into `Vm`
      num_krylov_vecs: Number of krylov iterations, should be identical to
        `Vm.shape[0] + 1`
      tol: Convergence parameter. Iteration is terminated if the norm of a
        krylov-vector falls below `tol`.
    Returns:
      jax.ShapedArray: An array of shape 
        `(num_krylov_vecs, np.prod(initial_state.shape))` of krylov vectors.
      jax.ShapedArray: The diagonal elements of the tridiagonal reduced
        operator ("alphas")
      jax.ShapedArray: The lower-diagonal elements of the tridiagonal reduced
        operator ("betas")
      jax.ShapedArray: The unnormalized residual of the Lanczos process.
      float: The norm of the residual.
      int: The number of performed iterations.
      bool: if `True`: iteration hit an invariant subspace.
            if `False`: iteration terminated without encountering
            an invariant subspace.
    """
    shape = (v0.size()[0],)
    Z = v0.norm()
    #only normalize if norm > tol, else return zero vector
    v = cond(Z > tol, lambda x: v0 / Z, lambda x: v0 * 0.0, None)
    Vm[start, :] = v.ravel()

    def set_betas(dummy):
        betas[start - 1] = Z
        return betas

    betas = cond(start > 0, set_betas, lambda x: betas, start)
    # body of the arnoldi iteration
    def body(vals):
        Vm, alphas, betas, previous_vector, _, i = vals
        Av = matvec(previous_vector, *args)
        Av, overlaps = iterative_classical_gram_schmidt(Av.ravel(), (i >= torch.arange(Vm.size(0), device = device))[:, None] * Vm, precision)
        
        alphas[i] = overlaps[i]
        norm = Av.norm()
        Av = Av.reshape(shape)
        # only normalize if norm is larger than threshold,
        # otherwise return zero vector
        Av = cond(norm > tol, lambda x: Av/norm, lambda x: Av * 0.0, None)

        def set_Vm_betas(dummy):
            Vm[i + 1, :] = Av.ravel()
            betas[i] = norm

            return (Vm, betas)

        Vm, betas = cond(i < num_krylov_vecs - 1, set_Vm_betas, lambda x: (Vm, betas), None)
        return [Vm, alphas, betas, Av, norm, i + 1]
    
    def cond_fun(vals):
        # Continue loop while iteration < num_krylov_vecs and norm > tol
        norm, iteration = vals[4], vals[5]
        counter_done = (iteration >= num_krylov_vecs)
        norm_not_too_small = norm > tol
        continue_iteration = cond(counter_done, lambda x: False, lambda x: norm_not_too_small, None)
        return continue_iteration

    initial_values = [Vm, alphas, betas, v, Z, start]
    final_values = while_loop(cond_fun, body, initial_values)
    Vm, alphas, betas, residual, norm, it = final_values
    return Vm, alphas, betas, residual, norm, it, norm < tol


    



def implicitly_restarted_lanczos_method(matvec, args, initial_state, num_krylov_vecs, numeig, which, tol, maxiter, precision, device = "cuda"):
    """
    Implicitly restarted lanczos factorization of `matvec`. The routine
    finds the lowest `numeig` eigenvector-eigenvalue pairs of `matvec`
    by alternating between compression and re-expansion of an initial
    `num_krylov_vecs`-step Lanczos factorization.
    Note: The caller has to ensure that the dtype of the return value
    of `matvec` matches the dtype of the initial state. Otherwise jax
    will raise a TypeError.
    NOTE: Under certain circumstances, the routine can return spurious
    eigenvalues 0.0: if the Lanczos iteration terminated early
    (after numits < num_krylov_vecs iterations)
    and numeig > numits, then spurious 0.0 eigenvalues will be returned.
    References:
    http://emis.impa.br/EMIS/journals/ETNA/vol.2.1994/pp1-21.dir/pp1-21.pdf
    http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf
    Args:
      matvec: A callable representing the linear operator.
      args: Arguments to `matvec`.  `matvec` is called with
        `matvec(x, *args)` with `x` the input array on which
        `matvec` should act.
      initial_state: An starting vector for the iteration.
      num_krylov_vecs: Number of krylov vectors of the lanczos factorization.
        numeig: The number of desired eigenvector-eigenvalue pairs.
      which: Which eigenvalues to target.
        Currently supported: `which = 'LR'` (largest real part).
      tol: Convergence flag. If the norm of a krylov vector drops below `tol`
        the iteration is terminated.
      maxiter: Maximum number of (outer) iteration steps.
      precision: jax.lax.Precision used within lax operations.
    Returns:
      jax.ShapedArray: Eigenvalues
      List: Eigenvectors
      int: Number of inner krylov iterations of the last lanczos
        factorization.
    """

    if args is None:
      args = (1,)

    def check_eigvals_convergence(beta_m, Hm, Hm_norm, tol):

        eigvals, eigvecs = torch.linalg.eigh(Hm)
        # TODO (mganahl) confirm that this is a valid matrix norm)
        thresh = torch.maximum(torch.finfo(eigvals.dtype).eps * Hm_norm, torch.abs(eigvals) * tol)
        vals = torch.abs(eigvecs[-1, :])

        return torch.all(beta_m * vals < thresh)

    def get_vectors(Vm, unitary, inds, numeig):

        def body_vector(i, states):
            dim = unitary.size(1)
            n, m = np.divmod(i, dim)
            states[n].index_add_(0, torch.arange(states.size(1), device = device), Vm[m, :] * unitary[m, inds[n]])
            return states

        state_vectors = torch.zeros([numeig, Vm.size(1)], dtype=Vm.dtype, device = device)
        state_vectors = fori_loop(0, numeig * Vm.size(0), body_vector, state_vectors)
        state_norms = torch.linalg.norm(state_vectors, axis=1)
        state_vectors = state_vectors / state_norms[:, None]

        return state_vectors


    shape = (torch.numel(initial_state),)
    dtype = initial_state.dtype

    dim = torch.numel(initial_state)
    num_expand = num_krylov_vecs - numeig
    #note: the second part of the cond is for testing purposes
    if num_krylov_vecs <= numeig < dim:
      raise ValueError(f"num_krylov_vecs must be between numeig <"
                       f" num_krylov_vecs <= dim = {dim},"
                       f" num_krylov_vecs = {num_krylov_vecs}")
    if numeig > dim:
      raise ValueError(f"number of requested eigenvalues numeig = {numeig} "
                       f"is larger than the dimension of the operator "
                       f"dim = {dim}")

    # initialize arrays
    Vm = torch.zeros((num_krylov_vecs, torch.numel(initial_state.ravel())), dtype=dtype, device = device)
    alphas = torch.zeros(num_krylov_vecs, dtype=dtype, device = device)
    betas = torch.zeros(num_krylov_vecs - 1, dtype=dtype, device = device)

    # perform initial lanczos factorization
    lanczos_fact = _lanczos_fact
    Vm, alphas, betas, residual, norm, numits, ar_converged = lanczos_fact(
        matvec, args, initial_state, Vm, alphas, betas, 0, num_krylov_vecs, tol,
        precision)
    fm = residual.ravel() * norm
    # generate needed functions
    #check_eigvals_convergence = _check_eigvals_convergence_eigh(jax)
    #get_vectors = _get_vectors(jax)

    # sort_fun returns `num_expand` least relevant eigenvalues
    # (those to be projected out)
    #if which == 'LA':
    #  sort_fun = jax.tree_util.Partial(_LA_sort(jax), num_expand)
    #elif which == 'SA':
    #  sort_fun = jax.tree_util.Partial(_SA_sort(jax), num_expand)
    #elif which == 'LM':
    #  sort_fun = jax.tree_util.Partial(_LM_sort(jax), num_expand)
    #else:
    #  raise ValueError(f"which = {which} not implemented")

    if which == "SA":
        sort_fun = lambda eigvals: torch.argsort(eigvals)
    
    else:
        raise ValueError("argument 'which' must be SA")

    it = 1  # we already did one lanczos factorization
    def outer_loop(carry):
        alphas, betas, Vm, fm, it, numits, ar_converged, _, _, = carry
        # pack into alphas and betas into tridiagonal matrix
        Hm = torch.diag(alphas) + torch.diag(betas, -1) + torch.diag(betas.conj(), 1)
        evals, _ = torch.linalg.eigh(Hm)
        shifts = sort_fun(evals)
        # perform shifted QR iterations to compress lanczos factorization
        # Note that ||fk|| typically decreases as one iterates the outer loop
        # indicating that iram converges.
        # ||fk|| = \beta_m in reference above
        Vk, Hk, fk = shifted_QR(Vm, Hm, fm, shifts, numeig, device = device)
        # extract new alphas and betas
        alphas = torch.diag(Hk)
        betas = torch.diag(Hk, -1)
        alphas[numeig:] = 0.0
        #alphas = alphas[numeig:]

        betas[numeig-1:] = 0.0
        #betas = betas[numeig-1:]

        beta_k = torch.linalg.norm(fk)
        Hktest = Hk[:numeig, :numeig]
        matnorm = torch.linalg.norm(Hktest)
        converged = check_eigvals_convergence(beta_k, Hktest, matnorm, tol)


        def do_lanczos(vals):
            Vk, alphas, betas, fk, _, _, _, _ = vals
            # restart
            Vm, alphas, betas, residual, norm, numits, ar_converged = lanczos_fact(
                matvec, args, torch.reshape(fk, shape), Vk, alphas, betas,
                numeig, num_krylov_vecs, tol, precision, device = device)
            fm = residual.ravel() * norm
            return [Vm, alphas, betas, fm, norm, numits, ar_converged, False]

        def cond_lanczos(vals):
            return vals[7]

        res = while_loop(cond_lanczos, do_lanczos, [
            Vk, alphas, betas, fk,
            torch.linalg.norm(fk), numeig, False,
            torch.logical_not(converged)
        ])

        Vm, alphas, betas, fm, norm, numits, ar_converged = res[0:7]

        out_vars = [
            alphas, betas, Vm, fm, it + 1, numits, ar_converged, converged, norm
        ]
        return out_vars

    def cond_fun(carry):
        it, ar_converged, converged = carry[4], carry[6], carry[7]
        return cond(
            it < maxiter, lambda x: x, lambda x: False,
            torch.logical_not(torch.logical_or(torch.tensor([converged]), torch.tensor([ar_converged]))))

    converged = False
    carry = [alphas, betas, Vm, fm, it, numits, ar_converged, converged, norm]
    res = while_loop(cond_fun, outer_loop, carry)
    alphas, betas, Vm = res[0], res[1], res[2]
    numits, ar_converged, converged = res[5], res[6], res[7]
    Hm = torch.diag(alphas) + torch.diag(betas, -1) + torch.diag(
        betas.conj(), 1)
    # FIXME (mganahl): under certain circumstances, the routine can still
    # return spurious 0 eigenvalues: if lanczos terminated early
    # (after numits < num_krylov_vecs iterations)
    # and numeig > numits, then spurious 0.0 eigenvalues will be returned
    Hm = (numits > torch.arange(num_krylov_vecs, device = device))[:, None] * Hm * (
        numits > torch.arange(num_krylov_vecs, device = device))[None, :]

    eigvals, U = torch.linalg.eigh(Hm)

    inds = sort_fun(eigvals)[:numeig]
    vectors = get_vectors(Vm, U, inds, numeig)
    return eigvals[inds], [
        torch.reshape(vectors[n, :], shape) for n in range(numeig)
    ], numits




















if __name__ == "__main__":
    from hamiltonian import Sky_phi
    from hamiltonian import ham_total
    from torch import profiler

    torch.cuda.set_device("cuda:0")

    L = 16
    B0 = 0.0
    Bext = 1.0
    dev = device("cuda:0")
    J_1 = -1.0

    center  = L / 2 - 0.5
    delta   = 0.5
    scalfac = 1.0
    B_0     = tensor([B0], dtype = torch.double).cuda()
    B_ext   = tensor([Bext], dtype = torch.double).cuda()
    phi_i   = tensor(Sky_phi(L, center, delta, scalfac), dtype = torch.double).cuda()

    if B_0.dtype == torch.double:
        H = ham_total(L, J_1 , B_0, B_ext, phi_i, prec = 64)
    else:
        H = ham_total(L, J_1 , B_0, B_ext, phi_i, prec = 32)

    H_linop = CsrLinOp(stack([H.storage._row, H.storage._col], dim = 0), H.storage._value, H.size(0))

    def matvec(v, dummy):

        return H_linop._mv(v)
        #return H.matmul(v)
    
    initial_state = torch.tensor([1.0]*(2**L), dtype = torch.double).random_().cuda()
    initial_state = initial_state / initial_state.norm()

    num_krylov_vecs = 20
    numeig = 3
    which = "SA"
    tol = 1e-8
    maxiter = 1000
    precision = 1e-10

    #with profiler.profile(activities=[ profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], with_stack = True, profile_memory = True) as prof:
    eigval,eigvec,num_it = implicitly_restarted_lanczos_method(matvec, None, initial_state, num_krylov_vecs, numeig, which, tol, maxiter, precision)
    #print(prof.key_averages().table(sort_by = "self_cuda_memory_usage", row_limit = 40))
    
    print(eigval)
    print(num_it)

    

