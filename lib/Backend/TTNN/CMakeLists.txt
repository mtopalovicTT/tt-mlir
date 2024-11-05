set(LIB_NAME TTNNWrapperLib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    TTNNWrapperLib.cpp
)

# add_mlir_library doenst work because it inheritcs LLVM disabled exceptions that doesnt compiler with tt-metal
# add_library(${LIB_NAME} SHARED ${SOURCES}) # missing mlir::MemRefType const& a tal
add_library(${LIB_NAME} STATIC ${SOURCES})

# Specify the include directories for the library
target_include_directories(${LIB_NAME} 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${PROJECT_SOURCE_DIR}/include/ttmlir/Backend/TTNN/)

target_include_directories(${LIB_NAME} PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(${LIB_NAME} PUBLIC TTNN_LIBRARY)
# target_link_libraries(${LIB_NAME} PUBLIC MLIRTTDialect)
# target_link_libraries(${LIB_NAME} 
#     PUBLIC
#     MLIRTTNNOpsIncGen
#     MLIRTTNNPassesIncGen 
#     MLIRTTOpsIncGen)

# Add TTNNWrapperLib to the export set
install(TARGETS ${LIB_NAME}
        EXPORT TTNNWrapperLibTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

# Export the targets
export(EXPORT TTNNWrapperLibTargets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/TTNNWrapperLibTargets.cmake"
       NAMESPACE TTNN::)