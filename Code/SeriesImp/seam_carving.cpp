// seam_carving.cpp

#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>        // For timing

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Custom image and matrix classes

class Img {
public:
    std::vector<uint32_t> pixels;
    int width, height, stride;

    Img(int w, int h)
        : pixels(w * h), width(w), height(h), stride(w) {}

    inline uint32_t& at(int row, int col) {
        assert(row >= 0 && row < height && col >= 0 && col < width);
        return pixels[row * stride + col];
    }
    inline const uint32_t& at(int row, int col) const {
        assert(row >= 0 && row < height && col >= 0 && col < width);
        return pixels[row * stride + col];
    }
};

class MatrixF {
public:
    std::vector<float> items;
    int width, height, stride;

    MatrixF(int w, int h)
        : items(w * h), width(w), height(h), stride(w) {}

    inline float& at(int row, int col) {
        assert(within(row, col));
        return items[row * stride + col];
    }
    inline const float& at(int row, int col) const {
        assert(within(row, col));
        return items[row * stride + col];
    }

    inline bool within(int row, int col) const {
        return (row >= 0 && row < height && col >= 0 && col < width);
    }
};

// Helper functions

static float rgb_to_lum(uint32_t bgra) {
    float b = ((bgra >>  (8*0)) & 0xFF) / 255.0f;
    float g = ((bgra >>  (8*1)) & 0xFF) / 255.0f;
    float r = ((bgra >>  (8*2)) & 0xFF) / 255.0f;
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

static void luminance(const Img &img, MatrixF &lum) {
    assert(img.width == lum.width && img.height == lum.height);
    for (int y = 0; y < lum.height; ++y)
        for (int x = 0; x < lum.width; ++x)
            lum.at(y, x) = rgb_to_lum(img.at(y, x));
}

static float sobel_filter_at(const MatrixF &mat, int cx, int cy) {
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
            int x = cx + dx, y = cy + dy;
            float c = mat.within(y, x) ? mat.at(y, x) : 0.0f;
            sx += c * gx[dy+1][dx+1];
            sy += c * gy[dy+1][dx+1];
        }
    }
    return sx*sx + sy*sy;
}

static void sobel_filter(const MatrixF &mat, MatrixF &grad) {
    assert(mat.width == grad.width && mat.height == grad.height);
    for (int y = 0; y < mat.height; ++y)
        for (int x = 0; x < mat.width; ++x)
            grad.at(y, x) = sobel_filter_at(mat, x, y);
}

// static void grad_to_dp(const MatrixF &grad, MatrixF &dp) {
//     assert(grad.width == dp.width && grad.height == dp.height);
//     int W = grad.width, H = grad.height;
//     for (int x = 0; x < W; ++x)
//         dp.at(0, x) = grad.at(0, x);

//     for (int y = 1; y < H; ++y) {
//         for (int cx = 0; cx < W; ++cx) {
//             float min_val = std::numeric_limits<float>::max();
//             for (int dx = -1; dx <= 1; ++dx) {
//                 int x = cx + dx;
//                 if (x >= 0 && x < W)
//                     min_val = std::min(min_val, dp.at(y-1, x));
//             }
//             dp.at(y, cx) = grad.at(y, cx) + min_val;
//         }
//     }
// }

static void grad_to_dp(const MatrixF &grad, MatrixF &dp) {
    assert(grad.width == dp.width && grad.height == dp.height);

    // Compute first row in parallel.
    for (int x = 0; x < grad.width; ++x)
        dp.at(0, x) = grad.at(0, x);

    // Process each subsequent row sequentially; inner loop parallelized.
    for (int y = 1; y < grad.height; ++y) {
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

static void img_remove_column_at_row(Img &img, int row, int col) {
    auto* p = img.pixels.data() + row*img.stride;
    std::move_backward(p + col + 1, p + img.width, p + img.width);
}

static void mat_remove_column_at_row(MatrixF &mat, int row, int col) {
    auto* p = mat.items.data() + row*mat.stride;
    std::move_backward(p + col + 1, p + mat.width, p + mat.width);
}

static void compute_seam(const MatrixF &dp, std::vector<int> &seam) {
    int H = dp.height, W = dp.width;
    seam.resize(H);
    // bottom-up
    seam[H-1] = std::min_element(&dp.at(H-1,0), &dp.at(H-1,0)+W) - &dp.at(H-1,0);
    for (int y = H-2; y >= 0; --y) {
        int px = seam[y+1];
        float best = dp.at(y, px);
        seam[y] = px;
        for (int dx=-1; dx<=1; ++dx) {
            int x = px + dx;
            if (x>=0 && x<W && dp.at(y,x) < best) {
                best = dp.at(y,x);
                seam[y] = x;
            }
        }
    }
}

static void markout_sobel_patches(MatrixF &grad, const std::vector<int> &seam) {
    for (int y = 0; y < grad.height; ++y) {
        int x = seam[y];
        for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
            int ny = y+dy, nx = x+dx;
            if (grad.within(ny,nx))
                reinterpret_cast<uint32_t&>(grad.at(ny,nx)) = 0xFFFFFFFF;
        }
    }
}

// Main with performance measurement

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>\n";
        return 1;
    }
    const std::string inputFile  = argv[1];
    const std::string outputFile = argv[2];

    // Load and convert image
    cv::Mat input = cv::imread(inputFile, cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        std::cerr << "ERROR: Could not load " << inputFile << "\n";
        return 1;
    }
    cv::Mat imgBGRA;
    if      (input.channels()==3) cv::cvtColor(input, imgBGRA, cv::COLOR_BGR2BGRA);
    else if (input.channels()==4) imgBGRA = input.clone();
    else {
        std::cerr<<"Unsupported channels\n"; return 1;
    }

    int W = imgBGRA.cols, H = imgBGRA.rows;
    Img img(W, H);
    if (imgBGRA.isContinuous()) {
        std::memcpy(img.pixels.data(), imgBGRA.data, W*H*4);
    } else {
        for (int y = 0; y < H; ++y)
            std::memcpy(img.pixels.data() + y*img.stride,
                        imgBGRA.ptr(y), W*4);
    }

    MatrixF lum(W,H), grad(W,H), dp(W,H);
    std::vector<int> seam(H);
    int seams_to_remove = W / 4;

    // Timing variables
    double t_lum = 0.0, t_sobel = 0.0, t_dp = 0.0, t_seam = 0.0;
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1) luminance
    {
        auto a = std::chrono::high_resolution_clock::now();
        luminance(img, lum);
        auto b = std::chrono::high_resolution_clock::now();
        t_lum = std::chrono::duration<double,std::milli>(b - a).count();
    }

    // 2) Sobel
    {
        auto a = std::chrono::high_resolution_clock::now();
        sobel_filter(lum, grad);
        auto b = std::chrono::high_resolution_clock::now();
        t_sobel = std::chrono::duration<double,std::milli>(b - a).count();
    }

    // 3) seam removal loop
    for (int i = 0; i < seams_to_remove; ++i) {
        auto a = std::chrono::high_resolution_clock::now();
        grad_to_dp(grad, dp);
        auto b = std::chrono::high_resolution_clock::now();
        t_dp += std::chrono::duration<double,std::milli>(b - a).count();

        auto c = std::chrono::high_resolution_clock::now();
        compute_seam(dp, seam);
        auto d = std::chrono::high_resolution_clock::now();
        t_seam += std::chrono::duration<double,std::milli>(d - c).count();

        markout_sobel_patches(grad, seam);
        for (int y = 0; y < H; ++y) {
            int x = seam[y];
            img_remove_column_at_row(img, y, x);
            mat_remove_column_at_row(lum, y, x);
            mat_remove_column_at_row(grad, y, x);
        }
        --W; --lum.width; --grad.width; --dp.width;

        for (int y = 0; y < H; ++y) {
            for (int x = seam[y]; x < grad.width; ++x)
                if (reinterpret_cast<uint32_t&>(grad.at(y,x))==0xFFFFFFFF)
                    grad.at(y,x) = sobel_filter_at(lum, x, y);
            for (int x = seam[y]-1; x >= 0; --x) {
                if (reinterpret_cast<uint32_t&>(grad.at(y,x))==0xFFFFFFFF)
                    grad.at(y,x) = sobel_filter_at(lum, x, y);
                else break;
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double t_total = std::chrono::duration<double,std::milli>(t_end - t_start).count();

    // write output
    cv::Mat output(H, W, CV_8UC4, img.pixels.data(), img.stride * sizeof(uint32_t));
    if (!cv::imwrite(outputFile, output)) {
        std::cerr << "ERROR: Could not save " << outputFile << "\n";
        return 1;
    }
    std::cout << "OK: Generated " << outputFile << "\n\n";

    // timing
    std::cout << "Summary Timing (ms):\n";
    std::cout << "  Luminance:      " << t_lum   << "\n";
    std::cout << "  Sobel filter:   " << t_sobel << "\n";
    std::cout << "  grad_to_dp:     " << t_dp    << "\n";
    std::cout << "  compute_seam:   " << t_seam  << "\n";
    std::cout << "  Total runtime:  " << t_total << "\n";

    return 0;
}
