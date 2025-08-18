#include "index_io.h"
#include <fstream>
#include <cstdint>

// Магические числа для файла
static constexpr char MAGIC[4] = {'P','L','D','B'};
static constexpr uint32_t VERSION = 1;

// Сохраняет индекс в бинарный файл
bool saveIndex(const std::string& path, const std::vector<IndexedDescriptor>& db) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.good()) return false;
    
    // Размерность дескриптора
    uint32_t dim = db.empty() ? 0u : static_cast<uint32_t>(db[0].descriptor.size());
    uint64_t count = static_cast<uint64_t>(db.size());
    
    // Записываем заголовок
    f.write(MAGIC, 4);
    f.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
    f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    // Записываем данные
    for (const auto& it : db) {
        int32_t x = static_cast<int32_t>(it.topLeft.x);
        int32_t y = static_cast<int32_t>(it.topLeft.y);
        int32_t augType = static_cast<int32_t>(it.augmentationType);
        
        f.write(reinterpret_cast<const char*>(&x), sizeof(x));
        f.write(reinterpret_cast<const char*>(&y), sizeof(y));
        f.write(reinterpret_cast<const char*>(&augType), sizeof(augType));
        
        if (!it.descriptor.empty()) {
            f.write(reinterpret_cast<const char*>(it.descriptor.data()), 
                   sizeof(float) * it.descriptor.size());
        }
    }
    
    return f.good();
}

// Загружает индекс из бинарного файла
bool loadIndex(const std::string& path, std::vector<IndexedDescriptor>& db, size_t expectedDim) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    
    // Читаем заголовок
    char magic[4];
    f.read(magic, 4);
    if (!f.good() || std::memcmp(magic, MAGIC, 4) != 0) return false;
    
    uint32_t ver = 0, dim = 0; 
    uint64_t count = 0;
    f.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    if (!f.good() || ver != VERSION) return false;
    if (expectedDim != 0 && dim != expectedDim) return false;
    
    // Читаем данные
    db.clear(); 
    db.reserve(static_cast<size_t>(count));
    
    for (uint64_t i = 0; i < count; ++i) {
        int32_t x = 0, y = 0, augType = 0;
        f.read(reinterpret_cast<char*>(&x), sizeof(x));
        f.read(reinterpret_cast<char*>(&y), sizeof(y));
        f.read(reinterpret_cast<char*>(&augType), sizeof(augType));
        
        IndexedDescriptor id; 
        id.topLeft = cv::Point(x, y);
        id.augmentationType = augType;
        id.descriptor.resize(dim);
        
        if (dim > 0) {
            f.read(reinterpret_cast<char*>(id.descriptor.data()), 
                   sizeof(float) * dim);
        }
        
        if (!f.good()) return false;
        db.push_back(std::move(id));
    }
    
    return true;
}


