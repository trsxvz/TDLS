#ifndef TDLS_CORE_ADAPTORS_HPP
#define TDLS_CORE_ADAPTORS_HPP



/// \file
/// \brief Generic adaptors: call the TiledLUpp solvers directly on dense
/// math objects (matrices, vectors, views) instead of raw pointers.
/// \author Tristan Chenaille
/// \copyright Copyright (C) 2026 CEA. Released under the
/// BSD 3-Clause License (see the LICENSE file).
///
/// The adaptors accept any object matching a small STRUCTURAL contract -
/// no external library is named or included. Three families are
/// recognized:
///
///   - contiguous fixed-size objects and views: expose data() and an
///     indexing_policy type whose extents are constexpr (this matches
///     tfel::math tmatrix / tvector / View, among others);
///   - element-strided views: expose their runtime stride either
///     through a data() member returning a (pointer, stride) pair (this
///     matches tfel::math StridedCoalescedView, i.e. the map_strided SoA
///     views) or through a stride() / getStride() member next to a plain
///     data();
///   - runtime-sized objects: their indexing policy carries its extents
///     as data members, so its default-constructed instance reports a
///     zero extent (this matches tfel::math matrix / vector). They also
///     expose getIndexingPolicy(), from which the dimension is read at
///     run time.
///
/// A fixed-size matrix resolves the compile-time TiledLUpp solver, with the
/// dimension deduced from the indexing policy and the residency
/// template booleans of the raw API inferred from the argument types
/// (contiguous object -> internal, strided view -> external), mixed
/// freely. A runtime-sized matrix resolves the runtime TiledLUpp solver,
/// every argument reduced to its (pointer, stride) pair. On that path
/// the extents cannot be checked at compile time: the matrix must be
/// square and every vector extent must match its dimension - unchecked
/// preconditions, exactly as in the raw API.
///
/// solve, solve_inplace, substitute and substitute_inplace also accept a
/// matrix-like right-hand side holding one right-hand side per column:
/// A X = B, every column solved against the one factorization through
/// the _block entry points of the raw API. Their template parameter W
/// cuts the substitution into passes of W columns (0, the default, means
/// all columns in one pass; the last pass takes the remainder).
///
/// Two families are rejected at compile time with an explicit message:
/// row-strided matrix views (sub-matrix views, whose stride between rows
/// is unrelated to the column count) cannot be expressed by the TiledLUpp solvers'
/// single-stride addressing, and gather views holding one pointer per
/// element expose no data() at all.
///
/// The detection can be overridden for exotic types by specializing
/// tdls::storage_traits.
///
/// The entry points currently dispatch to the TiledLUpp solver family;
/// a family selection parameter can join when another family lands.



#include <cstddef>
#include <type_traits>

#include <tdls/core/macros.hpp>
#include <tdls/solvers/tiled_lupp/solver_dynamic.hpp>
#include <tdls/solvers/tiled_lupp/solver_static.hpp>



namespace tdls {



namespace detail {

/// \brief Detects a data() member returning a pointer.
template<typename T, typename = void>
struct has_data : std::false_type {};
template<typename T>
struct has_data<T, std::void_t<decltype(std::declval<const T&>().data())>>
    : std::is_pointer<decltype(std::declval<const T&>().data())> {};

/// \brief Detects a data() member returning a (pointer, stride) pair -
/// the shape of strided views, whose data pointer must not be mistaken
/// for contiguous storage.
template<typename T, typename = void>
struct has_pair_data : std::false_type {};
template<typename T>
struct has_pair_data<T, std::void_t<decltype(std::declval<const T&>().data().first),
                                    decltype(std::declval<const T&>().data().second)>>
    : std::bool_constant<std::is_pointer_v<decltype(std::declval<const T&>().data().first)> &&
                         std::is_integral_v<decltype(std::declval<const T&>().data().second)>> {};

/// \brief Element pointer type of a dense object, const flavour.
template<typename T, bool = has_pair_data<T>::value>
struct const_data_pointer {
    //! \brief pointer type returned by data()
    using type = decltype(std::declval<const T&>().data());
};
/// \brief Element pointer type of a dense object whose data() returns a
/// (pointer, stride) pair, const flavour.
template<typename T>
struct const_data_pointer<T, true> {
    //! \brief pointer type of the first pair member of data()
    using type = decltype(std::declval<const T&>().data().first);
};

/// \brief Element pointer type of a dense object, mutable flavour.
template<typename T, bool = has_pair_data<T>::value>
struct mutable_data_pointer {
    //! \brief pointer type returned by data()
    using type = decltype(std::declval<T&>().data());
};
/// \brief Element pointer type of a dense object whose data() returns a
/// (pointer, stride) pair, mutable flavour.
template<typename T>
struct mutable_data_pointer<T, true> {
    //! \brief pointer type of the first pair member of data()
    using type = decltype(std::declval<T&>().data().first);
};

/// \brief Detects a stride() member.
template<typename T, typename = void>
struct has_stride : std::false_type {};
template<typename T>
struct has_stride<T, std::void_t<decltype(std::declval<const T&>().stride())>> : std::true_type {};

/// \brief Detects a getStride() member.
template<typename T, typename = void>
struct has_get_stride : std::false_type {};
template<typename T>
struct has_get_stride<T, std::void_t<decltype(std::declval<const T&>().getStride())>>
    : std::true_type {};

/// \brief Detects a nested indexing_policy type.
template<typename T, typename = void>
struct has_indexing_policy : std::false_type {};
template<typename T>
struct has_indexing_policy<T, std::void_t<typename T::indexing_policy>> : std::true_type {};

/// \brief A type is "dense" when it exposes both a data pointer and an
/// indexing policy - the structural contract of the adaptors.
template<typename T>
inline constexpr bool is_dense_v =
    (has_data<T>::value || has_pair_data<T>::value) && has_indexing_policy<T>::value;

} // namespace detail



/// \brief Storage description of a dense object: element type, extents,
/// element pointer and element stride.
///
/// The primary template covers, structurally, every type exposing data()
/// and an indexing_policy (see the file documentation). Specialize it for
/// types that do not fit the structural contract.
/// \tparam DenseType dense object type (without cv-qualifiers/references)
template<typename DenseType, typename = void>
struct storage_traits;

/// \brief Storage description of dense objects matching the structural
/// contract: a data() member (plain pointer or (pointer, stride) pair)
/// and an indexing_policy type with constexpr extents.
/// \tparam DenseType dense object type (without cv-qualifiers/references)
template<typename DenseType>
struct storage_traits<DenseType, std::enable_if_t<detail::is_dense_v<DenseType>>> {
    //! \brief indexing policy of the object
    using indexing_policy = typename DenseType::indexing_policy;
    //! \brief a shorthand for the indexing size type
    using policy_size_type = typename indexing_policy::size_type;
    //! \brief element type (without cv-qualifiers)
    using value_type = std::remove_cv_t<
        std::remove_pointer_t<typename detail::const_data_pointer<DenseType>::type>>;
    //! \brief true when data() yields a mutable pointer
    static constexpr bool is_mutable = !std::is_const_v<
        std::remove_pointer_t<typename detail::mutable_data_pointer<DenseType>::type>>;
    //! \brief number of indices (1 = vector-like, 2 = matrix-like)
    static constexpr int arity = static_cast<int>(indexing_policy::arity);
    static_assert(arity == 1 || arity == 2, "tdls adaptors: only vector-like (arity 1) and "
                                            "matrix-like (arity 2) objects are supported");
    //! \brief extent along the first dimension
    static constexpr int extent0 = static_cast<int>(indexing_policy{}.size(0));
    //! \brief extent along the second dimension (1 for vectors)
    static constexpr int extent1 = (arity == 2) ? static_cast<int>(indexing_policy{}.size(1)) : 1;

    //! \return the compile-time part of the element stride, read from the
    //! indexing policy (distance between two column-consecutive elements)
    static constexpr int compute_policy_stride() {
        if constexpr (arity == 1) {
            return static_cast<int>(indexing_policy{}.getIndex(policy_size_type(1)));
        } else {
            return static_cast<int>(
                indexing_policy{}.getIndex(policy_size_type(0), policy_size_type(1)));
        }
    }
    //! \brief compile-time part of the element stride
    static constexpr int policy_stride = compute_policy_stride();

    //! \return true when the distance between rows is exactly extent1
    //! times the distance between columns (vectors: always true)
    [[nodiscard]] static constexpr bool has_uniform_rows() {
        if constexpr (arity == 1) {
            return true;
        } else {
            return static_cast<int>(indexing_policy{}.getIndex(
                       policy_size_type(1), policy_size_type(0))) == extent1 * policy_stride;
        }
    }
    // Single-stride addressing requires uniform rows. Row-strided matrix
    // views (sub-matrix views) violate this and cannot be passed to the
    // solvers; solve the full matrix or use an element-strided
    // (coalesced) view instead.
    static_assert(has_uniform_rows(),
                  "tdls adaptors: row-strided matrix views (sub-matrix views) cannot "
                  "be expressed by the single-stride addressing of the TiledLUpp solvers");

    //! \brief true when the object carries a runtime element stride
    static constexpr bool has_runtime_stride = detail::has_pair_data<DenseType>::value ||
                                               detail::has_stride<DenseType>::value ||
                                               detail::has_get_stride<DenseType>::value;
    //! \brief true when the storage is compile-time contiguous: the raw
    //! API may then use the internal residency mode (direct indexing)
    static constexpr bool is_internal = (policy_stride == 1) && !has_runtime_stride;

    //! \brief true when the extents are only known at run time: the
    //! default-constructed indexing policy then reports a zero extent,
    //! the signature of policies carrying their sizes as data members
    static constexpr bool has_runtime_extents = (extent0 == 0);

    //! \return the run-time extent along the first dimension, read from
    //! the indexing policy instance of the object (only instantiated on
    //! the runtime-sized path)
    //! \param[in] o dense object
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int
    runtime_extent0(const DenseType& o) noexcept {
        return static_cast<int>(o.getIndexingPolicy().size(0));
    }

    //! \return the run-time extent along the second dimension, read from
    //! the indexing policy instance of the object (only instantiated on
    //! the runtime-sized path, for matrix-like right-hand-side blocks)
    //! \param[in] o dense object
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int
    runtime_extent1(const DenseType& o) noexcept {
        return static_cast<int>(o.getIndexingPolicy().size(1));
    }

    //! \return the run-time distance between two row-consecutive elements,
    //! read from the indexing policy instance of the object (only
    //! instantiated on the runtime-sized path); to be scaled by
    //! runtime_view_stride
    //! \param[in] o dense object
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int
    runtime_row_stride(const DenseType& o) noexcept {
        return static_cast<int>(
            o.getIndexingPolicy().getIndex(policy_size_type(1), policy_size_type(0)));
    }

    //! \return the run-time distance between two column-consecutive
    //! elements, read from the indexing policy instance of the object
    //! (only instantiated on the runtime-sized path); to be scaled by
    //! runtime_view_stride
    //! \param[in] o dense object
    [[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int
    runtime_col_stride(const DenseType& o) noexcept {
        return static_cast<int>(
            o.getIndexingPolicy().getIndex(policy_size_type(0), policy_size_type(1)));
    }

    //! \return the pointer to the first element (const-ness is erased;
    //! mutating entry points check is_mutable beforehand)
    //! \param[in] o dense object
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr value_type*
    pointer(const DenseType& o) noexcept {
        if constexpr (detail::has_pair_data<DenseType>::value) {
            return const_cast<value_type*>(static_cast<const value_type*>(o.data().first));
        } else {
            return const_cast<value_type*>(static_cast<const value_type*>(o.data()));
        }
    }

    //! \return the runtime stride of the view (1 when the object carries
    //! none): the factor scaling every distance of the indexing policy
    //! \param[in] o dense object
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int
    runtime_view_stride(const DenseType& o) noexcept {
        if constexpr (detail::has_pair_data<DenseType>::value) {
            return static_cast<int>(o.data().second);
        } else if constexpr (detail::has_stride<DenseType>::value) {
            return static_cast<int>(o.stride());
        } else if constexpr (detail::has_get_stride<DenseType>::value) {
            return static_cast<int>(o.getStride());
        } else {
            return 1;
        }
    }

    //! \return the total element stride (compile-time part times the
    //! runtime stride of the view, when any)
    //! \param[in] o dense object
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int stride(const DenseType& o) noexcept {
        return policy_stride * runtime_view_stride(o);
    }
};



namespace detail {

/// \brief Common compile-time context of the adaptor entry points:
/// resolves the scalar type, the dimension and the TiledLUpp solver from the
/// matrix argument.
/// \tparam UserConfig user configuration (void = TiledLUppDefaultConfig)
/// \tparam MatrixType dense matrix type
template<typename UserConfig, typename MatrixType>
struct adaptor_context {
    //! \brief storage description of the matrix
    using mtraits = storage_traits<std::remove_cv_t<MatrixType>>;
    //! \brief scalar type of the system
    using scalar = typename mtraits::value_type;
    static_assert(std::is_floating_point_v<scalar>,
                  "tdls adaptors: the element type must be float or double");
    static_assert(mtraits::arity == 2, "tdls adaptors: A must be matrix-like");
    //! \brief true when the matrix is runtime-sized (dynamic solver path)
    static constexpr bool runtime_sized = mtraits::has_runtime_extents;
    static_assert(runtime_sized || mtraits::extent0 == mtraits::extent1,
                  "tdls adaptors: A must be square");
    //! \brief system dimension (fixed-size path; zero on the runtime path)
    static constexpr int N = mtraits::extent0;
    //! \brief resolved configuration
    using config =
        std::conditional_t<std::is_void_v<UserConfig>, TiledLUppDefaultConfig<scalar>, UserConfig>;
    //! \brief resolved compile-time solver (never instantiated on the
    //! runtime path: the discarded if-constexpr branches keep it unused)
    using solver = TiledLUppSolverStatic<scalar, N, config>;
    //! \brief resolved runtime solver (never instantiated on the
    //! fixed-size path)
    using dynamic_solver = TiledLUppSolverDynamic<scalar, config>;
};

/// \brief Checks that a vector-like argument matches the system: arity 1,
/// extent N, same scalar type. The extent is only checkable when both
/// the matrix and the vector are fixed-size; when either side is
/// runtime-sized (N == 0, or a runtime-sized vector), the extent match
/// is an unchecked precondition, as everywhere in the raw API.
/// \tparam VectorType dense vector type
/// \tparam Scalar     scalar type of the system
/// \tparam N          system dimension (0 on the runtime path)
template<typename VectorType, typename Scalar, int N>
constexpr void check_vector() {
    using vtraits = storage_traits<std::remove_cv_t<VectorType>>;
    static_assert(vtraits::arity == 1, "tdls adaptors: expected a vector-like object");
    static_assert(N == 0 || vtraits::has_runtime_extents || vtraits::extent0 == N,
                  "tdls adaptors: vector extent does not match the system dimension");
    static_assert(std::is_same_v<typename vtraits::value_type, Scalar>,
                  "tdls adaptors: mixed scalar types");
}

/// \brief Checks that the right-hand side and the solution share one
/// residency and one compile-time stride: the raw API carries a single
/// rhs_stride for both. With runtime-strided views, both must be mapped
/// on the same batch (equal runtime strides), which is the caller's
/// responsibility.
/// \tparam RhsType      right-hand-side type
/// \tparam SolutionType solution type
template<typename RhsType, typename SolutionType>
constexpr void check_rhs_pair() {
    using bt = storage_traits<std::remove_cv_t<RhsType>>;
    using xt = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(bt::is_internal == xt::is_internal && bt::policy_stride == xt::policy_stride &&
                      bt::has_runtime_stride == xt::has_runtime_stride,
                  "tdls adaptors: b and x must share the same residency and "
                  "stride (the raw API carries a single rhs_stride for both)");
}

/// \brief Checks a matrix-like right-hand-side block pair (B, X): matching
/// scalar, matching extents where compile-time knowledge allows, one
/// shared layout. On any runtime-sized side the extents are unchecked
/// preconditions, as everywhere in the raw API: B and X must have N rows
/// and the same column count.
/// \tparam RhsType      right-hand-side block type
/// \tparam SolutionType solution block type
/// \tparam Scalar       scalar type of the system
/// \tparam N            system dimension (0 on the runtime path)
template<typename RhsType, typename SolutionType, typename Scalar, int N>
constexpr void check_rhs_block() {
    using bt = storage_traits<std::remove_cv_t<RhsType>>;
    using xt = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(xt::arity == 2, "tdls adaptors: a matrix-like right-hand side requires a "
                                  "matrix-like solution");
    static_assert(std::is_same_v<typename bt::value_type, Scalar> &&
                      std::is_same_v<typename xt::value_type, Scalar>,
                  "tdls adaptors: mixed scalar types");
    static_assert(bt::has_runtime_extents == xt::has_runtime_extents,
                  "tdls adaptors: B and X must both be fixed-size or both runtime-sized");
    static_assert(N == 0 || !bt::has_runtime_extents,
                  "tdls adaptors: a runtime-sized right-hand-side block requires a "
                  "runtime-sized matrix A");
    static_assert(N == 0 || bt::has_runtime_extents ||
                      (bt::extent0 == N && xt::extent0 == N && bt::extent1 == xt::extent1),
                  "tdls adaptors: B and X must have N rows and the same column count");
    static_assert(bt::policy_stride == xt::policy_stride &&
                      bt::has_runtime_stride == xt::has_runtime_stride,
                  "tdls adaptors: B and X must share the same layout (the raw API "
                  "carries a single stride pair for both)");
}

/// \brief Pivot argument unwrapping: accepts a raw int pointer/array
/// (treated as contiguous caller-local storage) or any dense int object.
/// \tparam PivotType pivot argument type (without references)
template<typename PivotType>
struct pivot_access {
    //! \brief true for a raw int pointer or int array
    static constexpr bool is_raw =
        std::is_pointer_v<std::decay_t<PivotType>> &&
        std::is_same_v<std::remove_cv_t<std::remove_pointer_t<std::decay_t<PivotType>>>, int>;
    //! \brief true for a dense int object
    static constexpr bool is_dense = detail::is_dense_v<std::remove_cv_t<PivotType>>;
    static_assert(is_raw || is_dense, "tdls adaptors: the pivot must be an int pointer/array or "
                                      "a dense int object");

    //! \brief true when the pivot argument may be written (non-const
    //! object, non-const pointee for raw pointers)
    static constexpr bool is_mutable = [] {
        if constexpr (is_raw) {
            return !std::is_const_v<std::remove_pointer_t<std::decay_t<PivotType>>>;
        } else {
            return !std::is_const_v<PivotType> &&
                   storage_traits<std::remove_cv_t<PivotType>>::is_mutable;
        }
    }();

    //! \brief true when the pivot storage is compile-time contiguous
    static constexpr bool is_internal = [] {
        if constexpr (is_raw) {
            return true;
        } else {
            return storage_traits<std::remove_cv_t<PivotType>>::is_internal;
        }
    }();

    //! \return the pivot element pointer
    //! \param[in] p pivot argument
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int* pointer(const PivotType& p) noexcept {
        if constexpr (is_raw) {
            return const_cast<int*>(static_cast<const int*>(p));
        } else {
            using pt = storage_traits<std::remove_cv_t<PivotType>>;
            static_assert(std::is_same_v<typename pt::value_type, int>,
                          "tdls adaptors: the pivot element type must be int");
            return pt::pointer(p);
        }
    }

    //! \return the pivot element stride
    //! \param[in] p pivot argument
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static constexpr int stride(const PivotType& p) noexcept {
        if constexpr (is_raw) {
            return 1;
        } else {
            return storage_traits<std::remove_cv_t<PivotType>>::stride(p);
        }
    }
};

/// \brief Shared engine of the matrix right-hand-side entry points:
/// substitution of a block of columns from a prior factorize, cut into
/// passes of W columns (W = 0: all columns in one pass). Blocks are
/// always routed through the external addressing of the raw API: the
/// register-block convention of its internal mode is not the row-major
/// layout of dense matrix objects.
/// \tparam Ctx     adaptor context of the entry point
/// \tparam W       columns per pass (0 = all at once)
/// \tparam inplace false: read B, write X; true: overwrite X in place
///                 (b is then ignored and may alias x)
template<typename Ctx, int W, bool inplace, typename MatrixType, typename PivotType,
         typename RhsType, typename SolutionType>
TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr void
substitute_block_dispatch(const MatrixType& A, const PivotType& piv, const RhsType& b,
                          SolutionType& x) {
    using mt = typename Ctx::mtraits;
    using pa = pivot_access<PivotType>;
    using bt = storage_traits<std::remove_cv_t<RhsType>>;
    using xt = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(W >= 0, "tdls adaptors: W must be a non-negative column count");
    if constexpr (Ctx::runtime_sized) {
        const int n    = mt::runtime_extent0(A);
        const int m    = bt::has_runtime_extents ? bt::runtime_extent1(b) : bt::extent1;
        const int rs   = bt::has_runtime_extents
                             ? xt::runtime_row_stride(x) * xt::runtime_view_stride(x)
                             : xt::extent1 * xt::stride(x);
        const int xcol = bt::has_runtime_extents
                             ? xt::runtime_col_stride(x) * xt::runtime_view_stride(x)
                             : xt::stride(x);
        for (int c0 = 0; c0 < m; c0 += (W == 0 ? m : W)) {
            const int cw = (W == 0 || W > m - c0) ? m - c0 : W;
            if constexpr (inplace) {
                Ctx::dynamic_solver::substitute_inplace_block(
                    n, cw, mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv),
                    xt::pointer(x) + unsigned(c0) * unsigned(xcol), rs, xcol);
            } else {
                Ctx::dynamic_solver::substitute_block(
                    n, cw, mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv),
                    bt::pointer(b) + unsigned(c0) * unsigned(xcol),
                    xt::pointer(x) + unsigned(c0) * unsigned(xcol), rs, xcol);
            }
        }
    } else {
        constexpr int M = bt::extent1;
        static_assert(M >= 1, "tdls adaptors: B must have at least one column");
        constexpr int C = (W == 0 || W > M) ? M : W;
        const int rs    = xt::extent1 * xt::stride(x);
        const int xcol  = xt::stride(x);
        for (int c = 0; c < M / C; ++c) {
            const unsigned off = unsigned(c) * unsigned(C) * unsigned(xcol);
            if constexpr (inplace) {
                Ctx::solver::template substitute_inplace_block<C, false, pa::is_internal,
                                                               mt::is_internal>(
                    mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv),
                    xt::pointer(x) + off, rs, xcol);
            } else {
                Ctx::solver::template substitute_block<C, false, pa::is_internal, mt::is_internal>(
                    mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv),
                    bt::pointer(b) + off, xt::pointer(x) + off, rs, xcol);
            }
        }
        if constexpr (M % C > 0) {
            const unsigned off = unsigned(M - M % C) * unsigned(xcol);
            if constexpr (inplace) {
                Ctx::solver::template substitute_inplace_block<M % C, false, pa::is_internal,
                                                               mt::is_internal>(
                    mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv),
                    xt::pointer(x) + off, rs, xcol);
            } else {
                Ctx::solver::template substitute_block<M % C, false, pa::is_internal,
                                                       mt::is_internal>(
                    mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv),
                    bt::pointer(b) + off, xt::pointer(x) + off, rs, xcol);
            }
        }
    }
}

} // namespace detail



/// \brief Factor a dense matrix object in place, A := P*L*U.
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \param[in,out] A   matrix-like object (factored in place)
/// \param[in,out] piv pivot storage: int pointer/array or dense int object
/// \return false on a singular matrix.
template<typename UserConfig = void, typename MatrixType, typename PivotType>
[[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr bool factorize(MatrixType& A,
                                                                         PivotType& piv) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    static_assert(mt::is_mutable, "tdls adaptors: factorize writes into A");
    static_assert(!std::is_const_v<MatrixType>, "tdls adaptors: A must not be const here "
                                                "(factorize writes it)");
    static_assert(pa::is_mutable, "tdls adaptors: the pivot must be mutable here "
                                  "(factorize writes it)");
    if constexpr (ctx::runtime_sized) {
        return ctx::dynamic_solver::factorize(mt::runtime_extent0(A), mt::pointer(A), mt::stride(A),
                                              pa::pointer(piv), pa::stride(piv));
    } else {
        return ctx::solver::template factorize<pa::is_internal, mt::is_internal>(
            mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv));
    }
}

/// \brief Solve A x = b on dense objects: factorize + substitute.
///
/// With a matrix-like b (and x), all its columns are solved against the
/// one factorization: A X = B. W then cuts the substitution into passes
/// of W columns (0 = all columns in one pass; the last pass takes the
/// remainder). The forward pass is not folded into the factorization on
/// that path (see the raw solve_block).
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \tparam W          columns per substitution pass for a matrix-like b
///         (0 = all at once; must be 0 for a vector-like b)
/// \param[in,out] A   matrix-like object (factored in place)
/// \param[in,out] piv pivot storage: int pointer/array or dense int object
/// \param[in]     b   right-hand side, in original order: vector-like, or
///                matrix-like holding one right-hand side per column
/// \param[out]    x   solution, of the same shape as b
/// \return false on a singular matrix.
template<typename UserConfig = void, int W = 0, typename MatrixType, typename PivotType,
         typename RhsType, typename SolutionType>
[[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr bool
solve(MatrixType& A, PivotType& piv, const RhsType& b, SolutionType& x) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    using bt  = storage_traits<std::remove_cv_t<RhsType>>;
    using xt  = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(mt::is_mutable, "tdls adaptors: solve factors A in place");
    static_assert(xt::is_mutable, "tdls adaptors: solve writes into x");
    static_assert(!std::is_const_v<MatrixType> && !std::is_const_v<SolutionType>,
                  "tdls adaptors: A and x must not be const here (solve writes them)");
    static_assert(pa::is_mutable, "tdls adaptors: the pivot must be mutable here "
                                  "(solve writes it)");
    if constexpr (bt::arity == 2) {
        detail::check_rhs_block<RhsType, SolutionType, typename ctx::scalar, ctx::N>();
        if (!factorize<UserConfig>(A, piv)) return false;
        detail::substitute_block_dispatch<ctx, W, false>(A, piv, b, x);
        return true;
    } else {
        static_assert(W == 0, "tdls adaptors: W only applies to matrix-like right-hand sides");
        detail::check_vector<RhsType, typename ctx::scalar, ctx::N>();
        detail::check_vector<SolutionType, typename ctx::scalar, ctx::N>();
        detail::check_rhs_pair<RhsType, SolutionType>();
        if constexpr (ctx::runtime_sized) {
            return ctx::dynamic_solver::solve(mt::runtime_extent0(A), mt::pointer(A), mt::stride(A),
                                              pa::pointer(piv), pa::stride(piv), bt::pointer(b),
                                              xt::pointer(x), xt::stride(x));
        } else {
            return ctx::solver::template solve<xt::is_internal, pa::is_internal, mt::is_internal>(
                mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv), bt::pointer(b),
                xt::pointer(x), xt::stride(x));
        }
    }
}

/// \brief Solve A y = y on dense objects with the fused factorization
/// (forward substitution folded into factorize, backward pass after).
///
/// With a matrix-like y, all its columns are overwritten by the
/// solutions against the one factorization. W then cuts the substitution
/// into passes of W columns (0 = all columns in one pass; the last pass
/// takes the remainder). The forward pass is not folded into the
/// factorization on that path (see the raw solve_inplace_block).
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \tparam W          columns per substitution pass for a matrix-like y
///         (0 = all at once; must be 0 for a vector-like y)
/// \param[in,out] A   matrix-like object (factored in place)
/// \param[in,out] piv pivot storage: int pointer/array or dense int object
/// \param[in,out] y   right-hand side on entry, solution on exit:
///                vector-like, or matrix-like holding one right-hand
///                side per column
/// \return false on a singular matrix (y left partially updated with a
///         vector-like y, untouched with a matrix-like y).
template<typename UserConfig = void, int W = 0, typename MatrixType, typename PivotType,
         typename VectorType>
[[nodiscard]] TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr bool
solve_inplace(MatrixType& A, PivotType& piv, VectorType& y) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    using yt  = storage_traits<std::remove_cv_t<VectorType>>;
    static_assert(mt::is_mutable, "tdls adaptors: solve_inplace factors A in place");
    static_assert(yt::is_mutable, "tdls adaptors: solve_inplace writes into y");
    static_assert(!std::is_const_v<MatrixType> && !std::is_const_v<VectorType>,
                  "tdls adaptors: A and y must not be const here (solve_inplace writes them)");
    static_assert(pa::is_mutable, "tdls adaptors: the pivot must be mutable here "
                                  "(solve_inplace writes it)");
    if constexpr (yt::arity == 2) {
        detail::check_rhs_block<VectorType, VectorType, typename ctx::scalar, ctx::N>();
        if (!factorize<UserConfig>(A, piv)) return false;
        detail::substitute_block_dispatch<ctx, W, true>(A, piv, y, y);
        return true;
    } else {
        static_assert(W == 0, "tdls adaptors: W only applies to matrix-like right-hand sides");
        detail::check_vector<VectorType, typename ctx::scalar, ctx::N>();
        if constexpr (ctx::runtime_sized) {
            return ctx::dynamic_solver::solve_inplace(
                mt::runtime_extent0(A), mt::pointer(A), mt::stride(A), pa::pointer(piv),
                pa::stride(piv), yt::pointer(y), yt::stride(y));
        } else {
            return ctx::solver::template solve_inplace<yt::is_internal, pa::is_internal,
                                                       mt::is_internal>(
                mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv), yt::pointer(y),
                yt::stride(y));
        }
    }
}

/// \brief Solve x := U^-1 L^-1 P b on dense objects, from a prior
/// factorize. b and x must not alias.
///
/// With a matrix-like b (and x), every column of b is solved into the
/// matching column of x. W then cuts the substitution into passes of W
/// columns (0 = all columns in one pass; the last pass takes the
/// remainder).
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \tparam W          columns per substitution pass for a matrix-like b
///         (0 = all at once; must be 0 for a vector-like b)
/// \param[in]  A   factored matrix-like object
/// \param[in]  piv pivot storage produced by factorize
/// \param[in]  b   right-hand side, in original order: vector-like, or
///             matrix-like holding one right-hand side per column
/// \param[out] x   solution, of the same shape as b
template<typename UserConfig = void, int W = 0, typename MatrixType, typename PivotType,
         typename RhsType, typename SolutionType>
TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr void
substitute(const MatrixType& A, const PivotType& piv, const RhsType& b, SolutionType& x) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    using bt  = storage_traits<std::remove_cv_t<RhsType>>;
    using xt  = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(xt::is_mutable, "tdls adaptors: substitute writes into x");
    static_assert(!std::is_const_v<SolutionType>,
                  "tdls adaptors: x must not be const here (substitute writes it)");
    if constexpr (bt::arity == 2) {
        detail::check_rhs_block<RhsType, SolutionType, typename ctx::scalar, ctx::N>();
        detail::substitute_block_dispatch<ctx, W, false>(A, piv, b, x);
    } else {
        static_assert(W == 0, "tdls adaptors: W only applies to matrix-like right-hand sides");
        detail::check_vector<RhsType, typename ctx::scalar, ctx::N>();
        detail::check_vector<SolutionType, typename ctx::scalar, ctx::N>();
        detail::check_rhs_pair<RhsType, SolutionType>();
        if constexpr (ctx::runtime_sized) {
            ctx::dynamic_solver::substitute(mt::runtime_extent0(A), mt::pointer(A), mt::stride(A),
                                            pa::pointer(piv), pa::stride(piv), bt::pointer(b),
                                            xt::pointer(x), xt::stride(x));
        } else {
            ctx::solver::template substitute<xt::is_internal, pa::is_internal, mt::is_internal>(
                mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv), bt::pointer(b),
                xt::pointer(x), xt::stride(x));
        }
    }
}

/// \brief Solve in place on dense objects: x holds the unpermuted
/// right-hand side on entry and the solution on exit.
///
/// With a matrix-like x, every column is overwritten by its solution. W
/// then cuts the substitution into passes of W columns (0 = all columns
/// in one pass; the last pass takes the remainder).
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \tparam W          columns per substitution pass for a matrix-like x
///         (0 = all at once; must be 0 for a vector-like x)
/// \param[in]     A   factored matrix-like object
/// \param[in]     piv pivot storage produced by factorize
/// \param[in,out] x   right-hand side, then solution: vector-like, or
///                matrix-like holding one right-hand side per column
template<typename UserConfig = void, int W = 0, typename MatrixType, typename PivotType,
         typename SolutionType>
TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr void
substitute_inplace(const MatrixType& A, const PivotType& piv, SolutionType& x) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    using xt  = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(xt::is_mutable, "tdls adaptors: substitute_inplace writes into x");
    static_assert(!std::is_const_v<SolutionType>,
                  "tdls adaptors: x must not be const here (substitute_inplace writes it)");
    if constexpr (xt::arity == 2) {
        detail::check_rhs_block<SolutionType, SolutionType, typename ctx::scalar, ctx::N>();
        detail::substitute_block_dispatch<ctx, W, true>(A, piv, x, x);
    } else {
        static_assert(W == 0, "tdls adaptors: W only applies to matrix-like right-hand sides");
        detail::check_vector<SolutionType, typename ctx::scalar, ctx::N>();
        if constexpr (ctx::runtime_sized) {
            ctx::dynamic_solver::substitute_inplace(mt::runtime_extent0(A), mt::pointer(A),
                                                    mt::stride(A), pa::pointer(piv),
                                                    pa::stride(piv), xt::pointer(x), xt::stride(x));
        } else {
            ctx::solver::template substitute_inplace<xt::is_internal, pa::is_internal,
                                                     mt::is_internal>(
                mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv), xt::pointer(x),
                xt::stride(x));
        }
    }
}

/// \brief Solve A x = e_col on dense objects, from a prior factorize -
/// the consistent-tangent-operator path.
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \param[in]  A   factored matrix-like object
/// \param[in]  piv pivot storage produced by factorize
/// \param[in]  col index of the canonical column e_col
/// \param[out] x   vector-like solution
template<typename UserConfig = void, typename MatrixType, typename PivotType, typename SolutionType>
TDLS_HOST_DEVICE TDLS_FORCEINLINE constexpr void
substitute_canonical(const MatrixType& A, const PivotType& piv, const int col, SolutionType& x) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    detail::check_vector<SolutionType, typename ctx::scalar, ctx::N>();
    using xt = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(xt::is_mutable, "tdls adaptors: substitute_canonical writes into x");
    static_assert(!std::is_const_v<SolutionType>,
                  "tdls adaptors: x must not be const here (substitute_canonical writes it)");
    if constexpr (ctx::runtime_sized) {
        ctx::dynamic_solver::substitute_canonical(mt::runtime_extent0(A), mt::pointer(A),
                                                  mt::stride(A), pa::pointer(piv), pa::stride(piv),
                                                  col, xt::pointer(x), xt::stride(x));
    } else {
        ctx::solver::template substitute_canonical<xt::is_internal, pa::is_internal,
                                                   mt::is_internal>(
            mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv), col, xt::pointer(x),
            xt::stride(x));
    }
}



} // namespace tdls



#endif // TDLS_CORE_ADAPTORS_HPP
