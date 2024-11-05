add_mlir_dialect_library(MLIRTTNNDialect
        TTNNDialect.cpp
        TTNNOps.cpp
        TTNNOpsBackendInterfaces.cpp
        TTNNOpsTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTOpsIncGen
        TTNNWrapperLib

        LINK_LIBS PUBLIC
        TTMLIRTTNNUtils
        MLIRSCFToEmitC
        MLIRLinalgDialect
        MLIRMLProgramDialect
        TTNNWrapperLib
        )
