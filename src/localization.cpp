#include "localization.h"
#include <limits>
#include <cmath>

// Вычисляет L2 расстояние между двумя дескрипторами
double Localization::l2(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;  // квадрат разности
    }
    return std::sqrt(acc);  // корень из суммы квадратов
}

// Находит ближайший патч в базе данных
int Localization::findNearest(const std::vector<float>& query,
                               const std::vector<IndexedDescriptor>& database) {
    if (database.empty()) return -1;
    
    int bestIdx = -1;
    double bestDist = std::numeric_limits<double>::infinity();
    
    // Простой перебор: сравниваем с каждым патчем
    for (size_t i = 0; i < database.size(); ++i) {
        double d = l2(query, database[i].descriptor);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = static_cast<int>(i);
        }
    }
    
    return bestIdx;
}


