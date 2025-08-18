cmake_minimum_required(VERSION 3.15)
project(patch_localization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Поиск OpenCV
set(OpenCV_DIR "C:/Users/voron/opencv/build/x64/vc16/lib/cmake/opencv4/build")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV найден: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV библиотеки: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV не найден! Установите OpenCV 4.x")
endif()

# Опция для TinyDNN
option(USE_TINYDNN "Enable TinyDNN for neural network features" ON)

if(USE_TINYDNN)
    set(TINYDNN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tiny-dnn-master")
    if(EXISTS "${TINYDNN_INCLUDE_DIRS}/tiny_dnn/tiny_dnn.h")
        message(STATUS "TinyDNN найден: ${TINYDNN_INCLUDE_DIRS}")
        add_definitions(-DUSE_TINYDNN)
    else()
        message(WARNING "TinyDNN не найден, отключаем")
        set(USE_TINYDNN OFF)
    endif()
else()
    message(STATUS "TinyDNN отключен - используем только гистограммы")
endif()

# Создаем библиотеку core
add_library(core STATIC
    src/image_processor.cpp
    src/localization.cpp
    src/autoencoder.cpp
    src/index_io.cpp
)

target_include_directories(core
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(USE_TINYDNN)
    target_include_directories(core PUBLIC ${TINYDNN_INCLUDE_DIRS})
endif()

target_link_libraries(core
    PUBLIC
        ${OpenCV_LIBS}
)

# Создаем исполняемый файл
add_executable(patch_localization
    src/main.cpp
)

target_link_libraries(patch_localization
    PRIVATE core
)

# Настройки компилятора
if (MSVC)
    target_compile_options(core PRIVATE /W3)
    target_compile_options(patch_localization PRIVATE /W3)
    
    # TinyDNN требует /bigobj из-за большого количества шаблонов
    if(USE_TINYDNN)
        target_compile_options(core PRIVATE /bigobj)
        target_compile_options(patch_localization PRIVATE /bigobj)
    endif()
else()
    target_compile_options(core PRIVATE -Wall -Wextra)
    target_compile_options(patch_localization PRIVATE -Wall -Wextra)
endif()


