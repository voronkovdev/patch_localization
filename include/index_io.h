#pragma once

#include <string>
#include <vector>
#include "localization.h"

// Сохраняет/загружает базу данных дескрипторов в компактном бинарном формате
// Формат:
//  magic[4] = 'P','L','D','B'
//  uint32 version = 1
//  uint32 dim
//  uint64 count
//  повторяется count раз: int32 x, int32 y, float[dim]

// Сохраняет индекс в файл
bool saveIndex(const std::string& path, const std::vector<IndexedDescriptor>& db);

// Загружает индекс из файла
bool loadIndex(const std::string& path, std::vector<IndexedDescriptor>& db, size_t expectedDim = 256);


