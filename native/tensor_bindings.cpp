#include <pybind11/pybind11.h>
PYBIND11_MODULE(_tensor_native, m) {
    m.doc()="Differentiable C++ physics operators, registered with the Torch dispatcher";
    m.attr("api_version")=9;
}
