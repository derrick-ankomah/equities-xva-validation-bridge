#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// ---------- Matrix-vector multiply: y = A x  ----------
py::array_t<double> matvec(py::array_t<double> A, py::array_t<double> x) {
    py::buffer_info a = A.request(), xv = x.request();
    if (a.ndim != 2 || xv.ndim != 1) throw std::runtime_error("A must be 2D and x 1D");
    ssize_t n = a.shape[0], d = a.shape[1];          // buffer_info: use shape[.]
    if (xv.shape[0] != d) throw std::runtime_error("shape mismatch");
    auto Ar = A.unchecked<2>();
    auto xr = x.unchecked<1>();
    py::array_t<double> y(n);
    auto yr = y.mutable_unchecked<1>();
    for (ssize_t i=0;i<n;++i){
        double acc=0.0;
        for (ssize_t j=0;j<d;++j) acc += Ar(i,j)*xr(j);
        yr(i)=acc;                                   // unchecked: use (.)
    }
    return y;
}

// ---------- Apply Cholesky factor: x = mu + L z ----------
py::array_t<double> chol_draw(py::array_t<double> L, py::array_t<double> mu, size_t n_paths, uint64_t seed=42) {
    py::buffer_info lb = L.request(), mub = mu.request();
    if (lb.ndim != 2 || mub.ndim!=1) throw std::runtime_error("L must be 2D, mu 1D");
    ssize_t d = lb.shape[0];                         // buffer_info
    if (lb.shape[1] != d || mub.shape[0] != d) throw std::runtime_error("shape mismatch");

    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0,1.0);

    auto Lr = L.unchecked<2>();
    auto mur = mu.unchecked<1>();
    py::array_t<double> out({(ssize_t)n_paths, d});
    auto outw = out.mutable_unchecked<2>();

    std::vector<double> z(d);
    for (size_t k=0;k<n_paths;++k){
        for (ssize_t i=0;i<d;++i) z[i]=nd(gen);
        for (ssize_t i=0;i<d;++i){
            double acc = 0.0;
            for (ssize_t j=0;j<=i;++j) acc += Lr(i,j)*z[j];
            outw(k,i)=mur(i)+acc;
        }
    }
    return out;
}

// ---------- Low-rank SVD perturbation ----------
py::array_t<double> svd_perturb(py::array_t<double> U, py::array_t<double> S, py::array_t<double> x, double eps, int r) {
    auto Ub = U.request(); auto Sb = S.request(); auto xb = x.request();
    if (Ub.ndim!=2 || Sb.ndim!=1 || xb.ndim!=1) throw std::runtime_error("bad dims");
    ssize_t d = Ub.shape[0];                         // buffer_info
    if (Ub.shape[1] < r || Sb.shape[0] < r || xb.shape[0] != d) throw std::runtime_error("shape mismatch");
    auto Ur = U.unchecked<2>(); auto Sr = S.unchecked<1>(); auto xr = x.unchecked<1>();
    py::array_t<double> x2(d);
    auto x2w = x2.mutable_unchecked<1>();
    for (ssize_t i=0;i<d;++i) x2w(i)=xr(i);
    for (int k=0;k<r;++k){
        double scale = eps * Sr(k);                  // unchecked: shape()/operator()
        for (ssize_t i=0;i<d;++i){
            x2w(i) += scale * Ur(i,k);
        }
    }
    return x2;
}

// ---------- Bootstrap indices (B x n) ----------
py::array_t<long long> bootstrap_indices(ssize_t n, ssize_t B, uint64_t seed=123) {
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<long long> uid(0, n-1);
    py::array_t<long long> out({B, n});
    auto w = out.mutable_unchecked<2>();
    for (ssize_t b=0;b<B;++b)
        for (ssize_t i=0;i<n;++i)
            w(b,i) = uid(gen);
    return out;
}

// ---------- Residual group stats ----------
py::tuple residual_group_mean(py::array_t<double> y_true, py::array_t<double> y_pred, py::array_t<long long> group) {
    auto yt = y_true.unchecked<1>();
    auto yp = y_pred.unchecked<1>();
    auto g  = group.unchecked<1>();
    ssize_t n = yt.shape(0);                         // unchecked proxy: use shape(.)
    if (yp.shape(0)!=n || g.shape(0)!=n) throw std::runtime_error("length mismatch");

    std::vector<long long> ids; ids.reserve(n);
    for (ssize_t i=0;i<n;++i) ids.push_back(g(i));
    std::vector<long long> uniq = ids;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    std::vector<double> sum(uniq.size(), 0.0);
    std::vector<long long> cnt(uniq.size(), 0);

    for (ssize_t i=0;i<n;++i){
        long long id = g(i);
        auto it = std::lower_bound(uniq.begin(), uniq.end(), id);
        size_t idx = std::distance(uniq.begin(), it);
        sum[idx] += (yt(i)-yp(i));
        cnt[idx] += 1;
    }
    py::array_t<long long> out_ids(uniq.size());
    py::array_t<double> out_mean(uniq.size());
    auto oi = out_ids.mutable_unchecked<1>();
    auto om = out_mean.mutable_unchecked<1>();
    for (size_t k=0;k<uniq.size();++k){
        oi(k)=uniq[k];
        om(k)= sum[k] / std::max<long long>(1, cnt[k]);
    }
    return py::make_tuple(out_ids, out_mean);
}

PYBIND11_MODULE(fastops, m) {
    m.doc() = "Fast C++ ops for robustness/uncertainty/offset diagnostics";
    m.def("matvec", &matvec);
    m.def("chol_draw", &chol_draw, py::arg("L"), py::arg("mu"), py::arg("n_paths"), py::arg("seed")=42);
    m.def("svd_perturb", &svd_perturb, py::arg("U"), py::arg("S"), py::arg("x"), py::arg("eps"), py::arg("r"));
    m.def("bootstrap_indices", &bootstrap_indices, py::arg("n"), py::arg("B"), py::arg("seed")=123);
    m.def("residual_group_mean", &residual_group_mean);
}
