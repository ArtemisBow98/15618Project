// seam_carving.cpp

#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>        // OpenMP

//------------------------------------------------------------------------------
// Custom image and matrix classes
//------------------------------------------------------------------------------

class Img {
public:
    std::vector<uint32_t> pixels; // BGRA per-pixel
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

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

static float rgb_to_lum(uint32_t bgra) {
    float b = ((bgra >>  0) & 0xFF) / 255.0f;
    float g = ((bgra >>  8) & 0xFF) / 255.0f;
    float r = ((bgra >> 16) & 0xFF) / 255.0f;
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

//------------------------------------------------------------------------------
// Two-phase, triangular-tiled DP
//------------------------------------------------------------------------------
static void grad_to_dp(const MatrixF &grad, MatrixF &dp, int num_strips) {
    int H = grad.height, W = grad.width;
    int strip_h = (H + num_strips - 1) / num_strips;

    // first row
    #pragma omp parallel for
    for (int x = 0; x < W; ++x)
        dp.at(0, x) = grad.at(0, x);

    // each strip: two phases → 2 syncs per strip
    for (int s = 0; s < num_strips; ++s) {
        int y0 = 1 + s*strip_h;
        int y1 = std::min(H, y0 + strip_h);
        if (y0 >= y1) break;
        int h = y1 - y0;

        // Phase 1: downward-pointing triangles
        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < W + h - 1; ++t) {
            for (int i = 0; i < h; ++i) {
                int x = t - i;
                if (x < 0 || x >= W) continue;
                int y = y0 + i;
                float m = dp.at(y-1, x);
                if (x > 0)    m = std::min(m, dp.at(y-1, x-1));
                if (x+1 < W) m = std::min(m, dp.at(y-1, x+1));
                dp.at(y, x) = grad.at(y, x) + m;
            }
        }

        // Phase 2: upward-pointing triangles
        #pragma omp parallel for schedule(dynamic)
        for (int t = 1 - W; t < h; ++t) {
            for (int i = 0; i < h; ++i) {
                int x = t + i;
                if (x < 0 || x >= W) continue;
                int y = y0 + (h - 1) - i;
                float m = dp.at(y-1, x);
                if (x > 0)    m = std::min(m, dp.at(y-1, x-1));
                if (x+1 < W) m = std::min(m, dp.at(y-1, x+1));
                dp.at(y, x) = grad.at(y, x) + m;
            }
        }
    }
}

// Remove a pixel from image & matrix
static void img_remove_column_at_row(Img &img, int row, int col) {
    auto* p = img.pixels.data() + row*img.stride;
    std::move_backward(p + col + 1, p + img.width, p + img.width);
}
static void mat_remove_column_at_row(MatrixF &mat, int row, int col) {
    auto* p = mat.items.data() + row*mat.stride;
    std::move_backward(p + col + 1, p + mat.width, p + mat.width);
}

// Find seam & mark it
static void compute_seam(const MatrixF &dp, std::vector<int> &seam) {
    int H = dp.height, W = dp.width;
    seam.resize(H);
    // bottom row min
    seam[H-1] = std::distance(
        &dp.at(H-1,0),
        std::min_element(&dp.at(H-1,0), &dp.at(H-1,0) + W)
    );
    // backtrack
    for (int y = H-2; y >= 0; --y) {
        int px = seam[y+1];
        int best = px;
        float bestv = dp.at(y, px);
        for (int dx = -1; dx <= 1; ++dx) {
            int x = px + dx;
            if (x>=0 && x<W && dp.at(y,x) < bestv) {
                bestv = dp.at(y,x);
                best = x;
            }
        }
        seam[y] = best;
    }
}

static void markout_sobel_patches(MatrixF &grad, const std::vector<int> &seam) {
    for (int y = 0; y < grad.height; ++y) {
        int x = seam[y];
        for (int dy=-1; dy<=1; ++dy)
         for (int dx=-1; dx<=1; ++dx)
          if (grad.within(y+dy,x+dx))
            reinterpret_cast<uint32_t&>(grad.at(y+dy,x+dx)) = 0xFFFFFFFF;
    }
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [-proc <threads>] [-strips <N>] <in.png> <out.png>\n";
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    int argi = 1;
    int num_threads = omp_get_max_threads();
    int num_strips  = 4;

    if (argc > 1 && std::string(argv[argi]) == "-proc") {
        num_threads = std::atoi(argv[argi+1]);
        omp_set_num_threads(num_threads);
        argi += 2;
    }
    if (argc > argi && std::string(argv[argi]) == "-strips") {
        num_strips = std::atoi(argv[argi+1]);
        if (num_strips < 1) num_strips = 1;
        argi += 2;
    }
    if (argc - argi < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string inF  = argv[argi++];
    std::string outF = argv[argi++];

    std::cout << "Using " << num_threads
              << " threads, " << num_strips << " strips\n";

    // Load image
    cv::Mat in = cv::imread(inF, cv::IMREAD_UNCHANGED);
    if (in.empty()) { std::cerr<<"ERROR loading "<<inF<<"\n"; return 1; }

    cv::Mat bgra;
    if      (in.channels()==3) cv::cvtColor(in, bgra, cv::COLOR_BGR2BGRA);
    else if (in.channels()==4) bgra = in.clone();
    else { std::cerr<<"Unsupported channels\n"; return 1; }

    int W = bgra.cols, H = bgra.rows;
    Img img(W,H);
    if (bgra.isContinuous())
        std::memcpy(img.pixels.data(), bgra.data, W*H*4);
    else {
        for(int y=0;y<H;++y)
            std::memcpy(
              img.pixels.data()+y*img.stride,
              bgra.ptr(y),
              W*4
            );
    }

    MatrixF lum(W,H), grad(W,H), dp(W,H);
    std::vector<int> seam(H);
    int seams_to_remove = W/4;

    double t_lum=0, t_sobel=0, t_dp=0, t_seam=0;
    auto t0 = std::chrono::high_resolution_clock::now();
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        luminance(img, lum);
        auto t2 = std::chrono::high_resolution_clock::now();
        sobel_filter(lum, grad);
        auto t3 = std::chrono::high_resolution_clock::now();
        t_lum   = std::chrono::duration<double,std::milli>(t2-t1).count();
        t_sobel = std::chrono::duration<double,std::milli>(t3-t2).count();
    }

    for (int i = 0; i < seams_to_remove; ++i) {
        auto a = std::chrono::high_resolution_clock::now();
        grad_to_dp(grad, dp, num_strips);
        auto b = std::chrono::high_resolution_clock::now();
        t_dp += std::chrono::duration<double,std::milli>(b-a).count();

        auto c = std::chrono::high_resolution_clock::now();
        compute_seam(dp, seam);
        auto d = std::chrono::high_resolution_clock::now();
        t_seam += std::chrono::duration<double,std::milli>(d-c).count();

        markout_sobel_patches(grad, seam);
        for (int y = 0; y < H; ++y) {
            int x = seam[y];
            img_remove_column_at_row(img, y, x);
            mat_remove_column_at_row(lum, y, x);
            mat_remove_column_at_row(grad, y, x);
        }
        --W; --lum.width; --grad.width; --dp.width;

        // update around removed seam
        for (int y=0; y<H; ++y) {
            for (int x=seam[y]; x<W; ++x)
                if (reinterpret_cast<uint32_t&>(grad.at(y,x))==0xFFFFFFFF)
                    grad.at(y,x) = sobel_filter_at(lum, x, y);
            for (int x=seam[y]-1; x>=0; --x) {
                if (reinterpret_cast<uint32_t&>(grad.at(y,x))==0xFFFFFFFF)
                    grad.at(y,x) = sobel_filter_at(lum, x, y);
                else break;
            }
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    double t_total = std::chrono::duration<double,std::milli>(t4-t0).count();

    // write out
    cv::Mat out(H, W, CV_8UC4, img.pixels.data(), img.stride*4);
    if (!cv::imwrite(outF, out)) {
        std::cerr << "ERROR saving " << outF << "\n";
        return 1;
    }
    std::cout << "OK: wrote " << outF << "\n\n";
    std::cout << "Timing (ms):\n"
              << "  luminance:   " << t_lum   << "\n"
              << "  sobel:       " << t_sobel << "\n"
              << "  grad_to_dp:  " << t_dp    << "\n"
              << "  compute_seam:" << t_seam  << "\n"
              << "  total:       " << t_total << "\n";

    return 0;
}
