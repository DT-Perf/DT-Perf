#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <NumCpp.hpp>
#include <../include/new_dataframe.h> // 
#include <omp.h>

// 
nc::NdArray<double> standardize_data(const nc::NdArray<double> &X)
{
    nc::NdArray<double> mean = nc::mean(X, nc::Axis::COL);
    nc::NdArray<double> std_dev = nc::stdev(X, nc::Axis::COL);
    return (X - mean) / std_dev;
}

// 
std::vector<size_t> center_based_sampling(const nc::NdArray<double> &X, size_t num_samples)
{
    std::vector<size_t> sampled_indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, X.numRows() - 1);
    sampled_indices.push_back(dist(gen));

    nc::NdArray<double> center = X(sampled_indices[0], nc::Slice(0, X.numCols()));

    for (size_t i = 1; i < num_samples; ++i)
    {
        std::vector<size_t> remaining_indices;
        for (size_t j = 0; j < X.numRows(); ++j)
        {
            if (std::find(sampled_indices.begin(), sampled_indices.end(), j) == sampled_indices.end())
            {
                remaining_indices.push_back(j);
            }
        }

        nc::NdArray<double> distances(remaining_indices.size());
        
        #pragma omp parallel for
        for (size_t j = 0; j < remaining_indices.size(); ++j)
        {
            size_t idx = remaining_indices[j];
             // 
            distances[j] = nc::norm(X(idx, nc::Slice(0, X.numCols())) - center).item(); 

        }

        // nc::argmax 
        size_t max_index = nc::argmax(distances).item();
        size_t best_index = remaining_indices[max_index];
        sampled_indices.push_back(best_index);

        // 
        nc::NdArray<double> sampled_points(sampled_indices.size(), X.numCols());
        for (size_t k = 0; k < sampled_indices.size(); ++k)
        {
            sampled_points.put(k, {0, static_cast<int>(X.numCols())}, X(sampled_indices[k], nc::Slice(0, X.numCols())));
        }
        center = nc::mean(sampled_points, nc::Axis::ROW);
    }

    return sampled_indices;
}

// 
void process_and_select_samples(const std::string &file_path, size_t num_samples, const std::string &save_path)
{
    NewDataFrame df;
    df.read_csv(file_path);
    
    nc::NdArray<double> X = df.get_data();
    nc::NdArray<double> y = df.get_column(df.get_column_names().back());
    df.remove_column(df.get_column_names().back());

    nc::NdArray<double> X_scaled = standardize_data(X);

    std::vector<size_t> representative_indices = center_based_sampling(X_scaled, num_samples);

    nc::NdArray<double> X_representative(representative_indices.size(), X.numCols());
    nc::NdArray<double> y_representative(representative_indices.size(), 1);

    for (size_t i = 0; i < representative_indices.size(); ++i)
    {
        X_representative.put(i,{0,static_cast<int>(X.numCols())}, X(representative_indices[i], nc::Slice(0, X.numCols())));
        y_representative.put(i, 0, y(representative_indices[i],0));
    }

    NewDataFrame representative_df(X_representative, df.get_column_names());
    representative_df.add_column("target", y_representative);
    representative_df.write_csv(save_path);
}

PYBIND11_MODULE(represent_data, m)
{
    m.def("process_and_select_samples", &process_and_select_samples, "Process and select representative samples.");
}
