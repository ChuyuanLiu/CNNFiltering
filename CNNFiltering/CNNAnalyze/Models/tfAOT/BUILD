load('//tensorflow/compiler/aot:tfcompile.bzl', 'tf_library')

tf_library(
    name = 'graph',
    config = 'graph.config.pbtxt',
    cpp_class = 'tfAOT::Graph',
    graph = 'graph.pb',
)

cc_binary(
    name = 'libtfaot.so',
    srcs = ['tfAOT.cc','tfAOT.h'],
    deps = [':graph', '//third_party/eigen3'],
    linkopts = ['-lpthread'],
    linkshared = 1,
    copts = ['-fPIC'],
    linkstatic = True
)
