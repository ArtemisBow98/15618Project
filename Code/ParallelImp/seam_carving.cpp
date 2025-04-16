// seam_carving.cpp

#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>       // For timing measurements
#include <cstdlib>      // for atoi

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>        // Include OpenMP header

//------------------------------------------------------------------------------
// Custom image and matrix classes
//------------------------------------------------------------------------------

class Img {
public:
    std::vector<uint32_t> pixels; // Each pixel is 32-bit BGRA.
    int width, height, stride;

    Img(int w, int h)
        : pixels(w * h), width(w), height(h), stride(w) {}

    // Non-const version.
    inline uint32_t& at(int row, int col) {
        assert(row >= 0 && row < height && col >= 0 && col < width);
        return pixels[row * stride + col];
    }
    // Const version.
    inline const uint32_t& at(int row, int col) const {
        assert(row >= 0 && row < height && col < width);
        return pixels[row * stride + col];
    }
};

class MatrixF {
public:
    std::vector<float> items;
    int width, height, stride;

    MatrixF(int w, int h)
        : items(w * h), width(w), height(h), stride(w) {}

    // Non-const version.
    inline float& at(int row, int col) {
        assert(within(row, col));
        return items[row * stride + col];
    }
    // Const version.
    inline const float& at(int row, int col) const {
        assert(within(row, col));
        return items[row * stride + col];
    }

    inline bool within(int row, int col) const {
        return (col >= 0 && col < width && row >= 0 && row < height);
    }
};

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

// For BGRA data (little-endian), the bytes are arranged as Blue, Green, Red, Alpha.
// We extract red from bits 16–23.
static float rgb_to_lum(uint32_t bgra) {
    float b = ((bgra >> (8 * 0)) & 0xFF) / 255.0f;
    float g = ((bgra >> (8 * 1)) & 0xFF) / 255.0f;
    float r = ((bgra >> (8 * 2)) & 0xFF) / 255.0f;
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Compute the luminance matrix for the image.
static void luminance(const Img &img, MatrixF &lum) {
    assert(img.width == lum.width && img.height == lum.height);
    for (int y = 0; y < lum.height; ++y) {
        for (int x = 0; x < lum.width; ++x) {
            lum.at(y, x) = rgb_to_lum(img.at(y, x));
        }
    }
}

// Compute the squared magnitude of the Sobel filter response at (cx, cy).
static float sobel_filter_at(const MatrixF &mat, int cx, int cy) {
    // Sobel kernels.
    static float gx[3][3] = {
        {  1.0f,  0.0f, -1.0f },
        {  2.0f,  0.0f, -2.0f },
        {  1.0f,  0.0f, -1.0f }
    };
    static float gy[3][3] = {
        {  1.0f,  2.0f,  1.0f },
        {  0.0f,  0.0f,  0.0f },
        { -1.0f, -2.0f, -1.0f }
    };

    float sx = 0.0f, sy = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int x = cx + dx;
            int y = cy + dy;
            float c = mat.within(y, x) ? mat.at(y, x) : 0.0f;
            sx += c * gx[dy + 1][dx + 1];
            sy += c * gy[dy + 1][dx + 1];
        }
    }
    return sx * sx + sy * sy;
}

// Apply the Sobel filter over the entire matrix.
static void sobel_filter(const MatrixF &mat, MatrixF &grad) {
    assert(mat.width == grad.width && mat.height == grad.height);
    for (int y = 0; y < mat.height; ++y)
        for (int x = 0; x < mat.width; ++x)
            grad.at(y, x) = sobel_filter_at(mat, x, y);
}

// Create the dynamic programming (cumulative energy) matrix with OpenMP parallelization.
static void grad_to_dp(const MatrixF &grad, MatrixF &dp) {
    assert(grad.width == dp.width && grad.height == dp.height);

    // Compute first row in parallel.
    #pragma omp parallel for
    for (int x = 0; x < grad.width; ++x)
        dp.at(0, x) = grad.at(0, x);

    // Process each subsequent row sequentially; inner loop parallelized.
    for (int y = 1; y < grad.height; ++y) {
        #pragma omp parallel for schedule(static, 64)
        for (int cx = 0; cx < grad.width; ++cx) {
            float min_val = std::numeric_limits<float>::max();
            // Evaluate neighbors from the previous row.
            for (int dx = -1; dx <= 1; ++dx) {
                int x = cx + dx;
                if (x >= 0 && x < grad.width) {
                    float v = dp.at(y - 1, x);
                    if (v < min_val)
                        min_val = v;
                }
            }
            dp.at(y, cx) = grad.at(y, cx) + min_val;
        }
    }
}

// Remove one pixel (column) at the given row.
static void img_remove_column_at_row(Img &img, int row, int column) {
    auto* row_ptr = img.pixels.data() + row * img.stride;
    std::move_backward(row_ptr + column + 1, row_ptr + img.width, row_ptr + img.width);
}

// Remove one pixel (column) at the given row for the float matrix.
static void mat_remove_column_at_row(MatrixF &mat, int row, int column) {
    auto* row_ptr = mat.items.data() + row * mat.stride;
    std::move_backward(row_ptr + column + 1, row_ptr + mat.width, row_ptr + mat.width);
}

// Find the seam with the minimum cumulative energy.
static void compute_seam(const MatrixF &dp, std::vector<int> &seam) {
    int height = dp.height, width = dp.width;
    seam.resize(height);
    int y = height - 1;
    seam[y] = 0;
    for (int x = 1; x < width; ++x) {
        if (dp.at(y, x) < dp.at(y, seam[y]))
            seam[y] = x;
    }
    for (y = height - 2; y >= 0; --y) {
        seam[y] = seam[y + 1];
        for (int dx = -1; dx <= 1; ++dx) {
            int x = seam[y + 1] + dx;
            if (x >= 0 && x < width && dp.at(y, x) < dp.at(y, seam[y]))
                seam[y] = x;
        }
    }
}

// Mark a 3x3 patch around each seam pixel in the gradient matrix.
static void markout_sobel_patches(MatrixF &grad, const std::vector<int> &seam) {
    for (int y = 0; y < grad.height; ++y) {
        int x = seam[y];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx, ny = y + dy;
                if (grad.within(ny, nx))
                    reinterpret_cast<uint32_t&>(grad.at(ny, nx)) = 0xFFFFFFFF;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Main function using OpenCV for image I/O and modern C++ for argument handling
//------------------------------------------------------------------------------

// Usage message updated to include the optional -proc argument.
static void print_usage(const char* progname) {
    std::cout << "Usage: " << progname << " [-proc <num_threads>] <input_image> <output_image>\n";
}

int main(int argc, char* argv[]) {
    // Default values.
    int arg_index = 1;
    int num_threads = omp_get_max_threads(); // Default to max available.

    // Parse optional -proc argument.
    if (argc > 1 && std::string(argv[arg_index]) == "-proc") {
        if (argc < 5) {
            print_usage(argv[0]);
            return 1;
        }
        num_threads = std::atoi(argv[arg_index + 1]);
        omp_set_num_threads(num_threads);
        arg_index += 2;
    }

    if (argc - arg_index < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string inputFile = argv[arg_index];
    const std::string outputFile = argv[arg_index + 1];

    // Inform user of thread count.
    std::cout << "Using " << num_threads << " OpenMP thread(s)" << std::endl;

    // Load the input image using OpenCV.
    cv::Mat input = cv::imread(inputFile, cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        std::cerr << "ERROR: Could not load " << inputFile << "\n";
        return 1;
    }

    // Convert to 4-channel BGRA.
    cv::Mat imgBGRA;
    if (input.channels() == 3)
        cv::cvtColor(input, imgBGRA, cv::COLOR_BGR2BGRA);
    else if (input.channels() == 4)
        imgBGRA = input.clone();
    else {
        std::cerr << "ERROR: Unsupported number of channels (" << input.channels() << ")\n";
        return 1;
    }

    const int width = imgBGRA.cols;
    const int height = imgBGRA.rows;

    // Create our Img object and copy pixel data.
    Img img(width, height);
    if (imgBGRA.isContinuous())
        std::copy(imgBGRA.data, imgBGRA.data + width * height * imgBGRA.elemSize(),
                  reinterpret_cast<unsigned char*>(img.pixels.data()));
    else {
        for (int y = 0; y < height; ++y)
            std::copy(imgBGRA.ptr(y), imgBGRA.ptr(y) + width * imgBGRA.elemSize(),
                      reinterpret_cast<unsigned char*>(img.pixels.data() + y * img.stride));
    }

    // Create luminance, gradient, and dp matrices.
    MatrixF lum(width, height);
    MatrixF grad(width, height);
    MatrixF dp(width, height);
    std::vector<int> seam(height);

    // Set number of seams to remove (adjust this as needed).
    int seams_to_remove = img.width / 4;

    // Timing accumulators for different functions.
    double total_grad_to_dp_time = 0.0;
    double total_compute_seam_time = 0.0;

    // Time the computation of the luminance matrix.
    auto start_total = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    luminance(img, lum);
    auto end = std::chrono::high_resolution_clock::now();
    double luminance_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Time the Sobel filter computation.
    start = std::chrono::high_resolution_clock::now();
    sobel_filter(lum, grad);
    end = std::chrono::high_resolution_clock::now();
    double sobel_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Seam removal loop.
    for (int i = 0; i < seams_to_remove; ++i) {
        // Accumulate time for grad_to_dp.
        start = std::chrono::high_resolution_clock::now();
        grad_to_dp(grad, dp);
        end = std::chrono::high_resolution_clock::now();
        total_grad_to_dp_time += std::chrono::duration<double, std::milli>(end - start).count();

        // Accumulate time for compute_seam.
        start = std::chrono::high_resolution_clock::now();
        compute_seam(dp, seam);
        end = std::chrono::high_resolution_clock::now();
        total_compute_seam_time += std::chrono::duration<double, std::milli>(end - start).count();

        markout_sobel_patches(grad, seam);

        for (int y = 0; y < img.height; ++y) {
            int x = seam[y];
            img_remove_column_at_row(img, y, x);
            mat_remove_column_at_row(lum, y, x);
            mat_remove_column_at_row(grad, y, x);
        }

        --img.width;
        --lum.width;
        --grad.width;
        --dp.width;

        // Update gradient values in the region affected by the removed seam.
        for (int y = 0; y < grad.height; ++y) {
            for (int x = seam[y]; x < grad.width; ++x)
                if (reinterpret_cast<uint32_t&>(grad.at(y, x)) == 0xFFFFFFFF)
                    grad.at(y, x) = sobel_filter_at(lum, x, y);
            for (int x = seam[y] - 1; x >= 0; --x)
                if (reinterpret_cast<uint32_t&>(grad.at(y, x)) == 0xFFFFFFFF)
                    grad.at(y, x) = sobel_filter_at(lum, x, y);
                else
                    break;
        }
    }
    auto end_total = std::chrono::high_resolution_clock::now();
    double total_processing_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    // Create an OpenCV Mat header over the resulting image data.
    cv::Mat output(img.height, img.width, CV_8UC4, img.pixels.data(), img.stride * sizeof(uint32_t));
    if (!cv::imwrite(outputFile, output)) {
        std::cerr << "ERROR: Could not save file " << outputFile << "\n";
        return 1;
    }
    std::cout << "OK: Generated " << outputFile << "\n\n";

    // Summary timing output.
    std::cout << "Summary Timing (in milliseconds):\n";
    std::cout << "  Luminance:     " << luminance_time << " ms\n";
    std::cout << "  Sobel Filter:  " << sobel_time    << " ms\n";
    std::cout << "  Total grad_to_dp time (over " << seams_to_remove << " iterations): " 
              << total_grad_to_dp_time << " ms\n";
    std::cout << "  Total compute_seam time (over " << seams_to_remove << " iterations): " 
              << total_compute_seam_time << " ms\n";
    std::cout << "  Total processing time: " << total_processing_time << " ms\n";

    return 0;
}
