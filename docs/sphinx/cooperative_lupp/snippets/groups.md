# Groups of threads

A group of threads shares the rows of a system: with T threads, thread
`tx` holds rows `tx`, `tx + T` and so on. Every thread of the group
calls the same entry point, with its own rank `tx` and the same
workspace. The solver deduces the barrier of the group: the snippets
pass none, unless they say otherwise.

## On GPU

The group is made of lanes of one warp on NVIDIA, of one wavefront on
AMD: its natural place. {doc}`getting_started` solves one system with
a group of 2 lanes, and the GPU kernel of {doc}`batches_and_layouts`
puts several groups in each warp, one per system. Four rules hold:

- the group fits in a warp, 32 lanes on NVIDIA and 64 on AMD;
- its threads call the solver together, from the same path of the
  code;
- its workspace is shared by the group, in shared or global memory,
  and no other group uses it;
- the threads of the group know their rank `tx`, from 0 to
  `threads_per_system` - 1.

The solver checks them as far as it can. A group larger than a warp
is refused at compile time. On entry, an incomplete group or a
workspace private to each thread stops the program with a message,
before any exchange. Two groups on one workspace are caught only when
their lanes share a warp.

## On CPU

Two CPU threads, started and joined by the caller, solve the system
together. On CPU, the first two elements of the workspace must be zero
before the first call: they hold the barrier of the group. A group of
CPU threads works, but one thread per system is the fast choice on
CPU: the barrier between CPU threads costs more than a step of the
solve.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/group_cpu_threads.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The run_group helper

The next snippets start their groups of CPU threads with
`snippets::run_group`, the loop above in a function: one thread per
rank, the function run with the rank `tx`, then the wait. A group of
one thread runs on the calling thread.

```{literalinclude} ../../../snippets/common/group.hpp
:language: cpp
:start-after: // run_group begin
:end-before: // run_group end
```

## A runtime dimension

The number of threads and the size of the workspace, read from the
solver for a runtime n: 3 threads and 17 elements for n = 5, the last
thread holding a phantom row in its second slot.

$$
A = \begin{pmatrix}
5 & 1 & 0 & 0 & 1 \\
1 & 6 & 1 & 0 & 0 \\
0 & 1 & 7 & 1 & 0 \\
0 & 0 & 1 & 8 & 1 \\
1 & 0 & 0 & 1 & 9
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_2 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 12 \\ 16 \\ 27 \\ 40 \\ 50 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 5 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/group_runtime_dimension.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## The barrier in the exchanges of the caller

`make_sync` returns the barrier the solver deduces, for the exchanges
of the caller between two calls: here each thread writes the
right-hand side of the rows of the other one. Passed to the next call,
it serves as is.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
b_1 = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
b_2 = \begin{pmatrix} 7 \\ 7 \\ 8 \\ 10 \end{pmatrix}, \quad
x_1 = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}, \quad
x_2 = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/group_deduced_barrier.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## An explicit barrier

The same system with a barrier passed by the caller, here a barrier
of CPU threads, for the illustration. A barrier is needed only for a
group that the deduced barrier does not know: a group larger than a
warp on GPU, which takes a block barrier such as `__syncthreads`, or a
group of CPU threads in `long double`, whose atomics are not lock-free
on most CPUs. In both cases, the compiler says so. Any callable taking
no argument serves; the solver uses it as is, and the workspace then
needs no initial value.

$$
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
b = \begin{pmatrix} 14 \\ 14 \\ 24 \\ 33 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/group_explicit_barrier.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```

## A Newton iteration in a group

The MFront pattern on a small nonlinear system, run by the group. Each
thread builds the rows it holds, so the solver reads them without a
barrier. On return, every thread reads the whole Newton step and
updates its copy of x. A barrier keeps the next iteration from
overwriting the step before every thread has read it: the one
`make_sync` returns, the barrier the solver deduces. At the solution,
one factorization serves the tangent columns.

$$
F(x) = A x + x \odot x - c = 0, \quad
J(x) = A + 2 \operatorname{diag}(x), \quad
A = \begin{pmatrix}
4 & 1 & 0 & 2 \\
1 & 5 & 1 & 0 \\
0 & 1 & 6 & 1 \\
2 & 0 & 1 & 7
\end{pmatrix}
\begin{matrix} t_0 \\ t_1 \\ t_0 \\ t_1 \end{matrix}, \quad
c = \begin{pmatrix} 15 \\ 18 \\ 33 \\ 49 \end{pmatrix}, \quad
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
$$

```{literalinclude} ../../../snippets/cooperative_lupp/group_newton.cpp
:language: cpp
:start-after: // snippet begin
:end-before: // snippet end
:dedent: 4
```
