#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <NumCpp.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <../include/new_dataframe.h> // 假设你的NewDataFrame类放在这个头文件中
#include <omp.h>

// 标准化数据
nc::NdArray<double> standardize_data(const nc::NdArray<double> &X)
{
    nc::NdArray<double> mean = nc::mean(X, nc::Axis::ROW);
    nc::NdArray<double> std_dev = nc::stdev(X, nc::Axis::ROW);

    const double epsilon = 1e-8; // Small value to avoid division by zero
    return (X - mean) / (std_dev + epsilon);
}

std::vector<size_t> center_based_sampling(const nc::NdArray<double> &X, size_t num_samples)
{
    std::vector<size_t> sampled_indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, X.numRows() - 1);

    // 随机选择一个初始样本
    size_t initial_index = dist(gen);
    sampled_indices.push_back(initial_index);
    //std::cout << "Initial index: " << initial_index << std::endl;

    nc::NdArray<double> center = X(initial_index, nc::Slice(0, X.numCols()));
    //std::cout << "Initial center point: " << center << std::endl;

    // 进行采样
    for (size_t i = 1; i < num_samples; ++i)
    {
        std::vector<size_t> remaining_indices;

        // 获取剩余样本索引
        for (size_t j = 0; j < X.numRows(); ++j)
        {
            if (std::find(sampled_indices.begin(), sampled_indices.end(), j) == sampled_indices.end())
            {
                remaining_indices.push_back(j);
            }
        }

        //std::cout << "Remaining indices size: " << remaining_indices.size() << std::endl;
        // 假设 distances 是一个 std::vector<double>
        std::vector<double> distances(remaining_indices.size());

        // 计算剩余样本与当前中心的距离
        #pragma omp parallel for
        for (size_t j = 0; j < remaining_indices.size(); ++j)
        {
            size_t idx = remaining_indices[j];
            distances[j] = nc::norm(X(idx, nc::Slice(0, X.numCols())) - center).item(); // 获取标量值
        }

        // 找到最大值的索引
        auto max_iter = std::max_element(distances.begin(), distances.end());
        size_t max_index = std::distance(distances.begin(), max_iter);

        // 使用最大值的索引获取最佳索引
        size_t best_index = remaining_indices[max_index];

        //std::cout << "Selected best index: " << best_index << " with distance: " << distances[max_index] << std::endl;

        sampled_indices.push_back(best_index);

        // 更新中心点为已选样本的均值
        nc::NdArray<double> sampled_points(sampled_indices.size(), X.numCols());
        for (size_t k = 0; k < sampled_indices.size(); ++k)
        {
            sampled_points.put(k, {0, static_cast<int>(X.numCols())}, X(sampled_indices[k], nc::Slice(0, X.numCols())));
        }

        center = nc::mean(sampled_points, nc::Axis::ROW);
        //std::cout << "Updated center: " << center << std::endl;
    }

    //std::cout << "Sampling completed." << std::endl;
    return sampled_indices;
}

// 处理并选择代表性样本
void process_and_select_samples(const std::string &file_path, size_t num_samples, const std::string &save_path)
{
    double start_time, end_time;
    start_time = omp_get_wtime();
    
    NewDataFrame df;
    df.read_csv(file_path);

    nc::NdArray<double> X = df.get_data();
    X = X({0, static_cast<int>(X.numRows())}, {0, static_cast<int>(X.numCols()) - 1});
    nc::NdArray<double> y = df.get_column(df.get_column_names().back());
    df.remove_column(df.get_column_names().back());

    // 标准化数据
    nc::NdArray<double> X_scaled = standardize_data(X);

    // 中心点采样
    std::vector<size_t> representative_indices = center_based_sampling(X_scaled, num_samples);

    nc::NdArray<double> X_representative(representative_indices.size(), X.numCols());
    nc::NdArray<double> y_representative(representative_indices.size(), 1);

    // 获取代表性样本和对应的目标值
    for (size_t i = 0; i < representative_indices.size(); ++i)
    {
        X_representative.put(i, {0, static_cast<int>(X.numCols())}, X(representative_indices[i], nc::Slice(0, X.numCols())));
        y_representative.put(i, 0, y(representative_indices[i], 0));
    }

    // 写入代表性样本到文件
    NewDataFrame representative_df(X_representative, df.get_column_names());
    representative_df.add_column("target", y_representative);
    end_time = omp_get_wtime();
    std::cout << "Time taken: " << end_time - start_time << " seconds." << std::endl;

    representative_df.write_csv(save_path);
}

PYBIND11_MODULE(represent_data, m)
{
    m.def("process_and_select_samples", &process_and_select_samples, "Process and select representative samples.");
}
