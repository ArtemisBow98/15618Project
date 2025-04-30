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

#include <omp.h>

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
        assert(row >= 0 && row < height && col >= 0 && col < width);
        return items[row * stride + col];
    }
    inline const float& at(int row, int col) const {
        assert(row >= 0 && row < height && col >= 0 && col < width);
        return items[row * stride + col];
    }
};

static float rgb_to_lum(uint32_t bgra) {
    float b = ((bgra >>  0) & 0xFF) / 255.f;
    float g = ((bgra >>  8) & 0xFF) / 255.f;
    float r = ((bgra >> 16) & 0xFF) / 255.f;
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

static void luminance(const Img &img, MatrixF &lum) {
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x)
            lum.at(y, x) = rgb_to_lum(img.at(y, x));
}

static float sobel_filter_at(const MatrixF &mat, int cx, int cy) {
    static float gx[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
    static float gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};
    float sx = 0.f, sy = 0.f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int x = cx + dx, y = cy + dy;
            float v = (y >= 0 && y < mat.height && x >= 0 && x < mat.width)
                        ? mat.at(y, x) : 0.f;
            sx += v * gx[dy+1][dx+1];
            sy += v * gy[dy+1][dx+1];
        }
    }
    return sx*sx + sy*sy;
}

static void sobel_filter(const MatrixF &mat, MatrixF &grad) {
    for (int y = 0; y < mat.height; ++y)
        for (int x = 0; x < mat.width; ++x)
            grad.at(y,x) = sobel_filter_at(mat, x, y);
}

static void grad_to_dp(const MatrixF &grad, MatrixF &dp, int base_w) {
    int W = grad.width, H = grad.height;
    if (base_w >= W) {
        #pragma omp parallel for
        for (int x = 0; x < W; ++x)
            dp.at(0,x) = grad.at(0,x);
        for (int y = 1; y < H; ++y) {
            #pragma omp parallel for schedule(static,64)
            for (int cx = 0; cx < W; ++cx) {
                float m = dp.at(y-1,cx);
                if (cx>0)      m = std::min(m, dp.at(y-1,cx-1));
                if (cx+1 < W) m = std::min(m, dp.at(y-1,cx+1));
                dp.at(y,cx) = grad.at(y,cx) + m;
            }
        }
        return;
    }

    // Compute first row once
    #pragma omp parallel for
    for (int x = 0; x < W; ++x)
        dp.at(0,x) = grad.at(0,x);

    // Tiled DP
    int tri_h      = base_w/4 + 1;
    int num_strips = (H + tri_h - 1) / tri_h;
    int num_down   = (W + base_w - 1) / base_w;

    for (int s = 0; s < num_strips; ++s) {
        int y0 = s * tri_h;
        int h  = std::min(tri_h, H - y0);

        #pragma omp parallel for schedule(static)
        for (int t = 0; t < num_down; ++t) {
            for (int r = (s==0 ? 1 : 0); r < h; ++r) {
                int y = y0 + r;
                int start = std::max(0, t*base_w + r);
                int end   = std::min(W-1, (t+1)*base_w - 1 - r);
                if (start > end) continue;
                for (int x = start; x <= end; ++x) {
                    float m = dp.at(y-1, x);
                    if (x>0)      m = std::min(m, dp.at(y-1, x-1));
                    if (x+1 < W) m = std::min(m, dp.at(y-1, x+1));
                    dp.at(y, x) = grad.at(y, x) + m;
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for (int u = 0; u <= num_down; ++u) {
            for (int r = 1; r < h; ++r) {
                int y = y0 + r;
                int start = (u==0 ? 0 : u*base_w - r);
                int end   = (u==num_down ? W-1 : u*base_w + r - 1);
                start = std::max(0, start);
                end   = std::min(W-1, end);
                if (start > end) continue;
                for (int x = start; x <= end; ++x) {
                    float m = dp.at(y-1, x);
                    if (x>0)      m = std::min(m, dp.at(y-1, x-1));
                    if (x+1 < W) m = std::min(m, dp.at(y-1, x+1));
                    dp.at(y, x) = grad.at(y, x) + m;
                }
            }
        }
    }
}


static void img_remove_column_at_row(Img &img,int r,int c) {
    auto *p = img.pixels.data() + r*img.stride;
    std::move_backward(p+c+1, p+img.width, p+img.width);
}
static void mat_remove_column_at_row(MatrixF &m,int r,int c) {
    auto *p = m.items.data() + r*m.stride;
    std::move_backward(p+c+1, p+m.width, p+m.width);
}

static void compute_seam(const MatrixF &dp, std::vector<int> &seam) {
    int H = dp.height, W = dp.width;
    seam.resize(H);
    seam[H-1] = std::min_element(&dp.at(H-1,0), &dp.at(H-1,0)+W) - &dp.at(H-1,0);
    for (int y = H-2; y >= 0; --y) {
        int px = seam[y+1]; float best = dp.at(y,px);
        seam[y] = px;
        for (int d = -1; d <= 1; ++d) {
            int x = px + d;
            if (x>=0 && x< W && dp.at(y,x)<best) {
                best=dp.at(y,x); seam[y]=x;
            }
        }
    }
}
static void markout_sobel_patches(MatrixF &g,const std::vector<int>& s) {
    for(int y=0;y<g.height;++y){int x=s[y];for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){int ny=y+dy,nx=x+dx;if(ny>=0&&ny<g.height&&nx>=0&&nx<g.width)reinterpret_cast<uint32_t&>(g.at(ny,nx))=0xFFFFFFFF;}}
}

static void print_usage(const char *prog) {
    std::cout<<"Usage: "<<prog<<" [-proc <threads>] [-width <base_w>] <in.png> <out.png>\n";
}

int main(int argc, char* argv[]) {
    int argi = 1;
    int num_threads = omp_get_max_threads();
    int base_w = 6;

    if (argc > 1 && std::string(argv[argi]) == "-proc") {
        num_threads = std::atoi(argv[argi + 1]);
        omp_set_num_threads(num_threads);
        argi += 2;
    }

    if (argc > argi && std::string(argv[argi]) == "-width") {
        base_w = std::atoi(argv[argi + 1]);
        base_w = std::max(1, base_w);
        argi += 2;
    }

    if (argc - argi < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string inF = argv[argi++];
    std::string outF = argv[argi++];
    std::cout << "Threads=" << num_threads << "  base_w=" << base_w << "\n";

    cv::Mat input = cv::imread(inF, cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        std::cerr << "ERROR loading " << inF << "\n";
        return 1;
    }

    cv::Mat bgra;
    if (input.channels() == 3) {
        cv::cvtColor(input, bgra, cv::COLOR_BGR2BGRA);
    } else if (input.channels() == 4) {
        bgra = input.clone();
    } else {
        std::cerr << "Bad channels\n";
        return 1;
    }

    int W = bgra.cols;
    int H = bgra.rows;
    Img img(W, H);
    if (bgra.isContinuous()) {
        std::memcpy(img.pixels.data(), bgra.data, W * H * 4);
    } else {
        for (int y = 0; y < H; ++y) {
            std::memcpy(img.pixels.data() + y * img.stride, bgra.ptr(y), W * 4);
        }
    }

    MatrixF lum(W, H), grad(W, H), dp(W, H);
    std::vector<int> seam(H);
    int seams_to_remove = W / 4;
    double luminance_time = 0, sobel_time = 0;
    double total_grad_to_dp_time = 0, total_compute_seam_time = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    {
        auto s = std::chrono::high_resolution_clock::now();
        luminance(img, lum);
        auto e = std::chrono::high_resolution_clock::now();
        luminance_time = std::chrono::duration<double, std::milli>(e - s).count();
    }

    {
        auto s = std::chrono::high_resolution_clock::now();
        sobel_filter(lum, grad);
        auto e = std::chrono::high_resolution_clock::now();
        sobel_time = std::chrono::duration<double, std::milli>(e - s).count();
    }

    for (int i = 0; i < seams_to_remove; ++i) {
        {
            auto s = std::chrono::high_resolution_clock::now();
            grad_to_dp(grad, dp, base_w);
            auto e = std::chrono::high_resolution_clock::now();
            total_grad_to_dp_time += std::chrono::duration<double, std::milli>(e - s).count();
        }
        {
            auto s = std::chrono::high_resolution_clock::now();
            compute_seam(dp, seam);
            auto e = std::chrono::high_resolution_clock::now();
            total_compute_seam_time += std::chrono::duration<double, std::milli>(e - s).count();
        }

        markout_sobel_patches(grad, seam);

        for (int y = 0; y < H; ++y) {
            int x = seam[y];
            img_remove_column_at_row(img, y, x);
            mat_remove_column_at_row(lum, y, x);
            mat_remove_column_at_row(grad, y, x);
        }
        --W;
        --lum.width;
        --grad.width;
        --dp.width;

        for (int y = 0; y < H; ++y) {
            for (int x = seam[y]; x < W; ++x) {
                if (reinterpret_cast<uint32_t&>(grad.at(y, x)) == 0xFFFFFFFF) {
                    grad.at(y, x) = sobel_filter_at(lum, x, y);
                }
            }
            for (int x = seam[y] - 1; x >= 0; --x) {
                if (reinterpret_cast<uint32_t&>(grad.at(y, x)) == 0xFFFFFFFF) {
                    grad.at(y, x) = sobel_filter_at(lum, x, y);
                } else {
                    break;
                }
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    cv::Mat out(H, W, CV_8UC4, img.pixels.data(), img.stride * 4);
    cv::imwrite(outF, out);

    std::cout << "\nSummary Timing (in milliseconds):\n";
    std::cout << "  Luminance:     " << luminance_time << " ms\n";
    std::cout << "  Sobel Filter:  " << sobel_time << " ms\n";
    std::cout << "  Total grad_to_dp time (over " << seams_to_remove << " iterations): " << total_grad_to_dp_time << " ms\n";
    std::cout << "  Total compute_seam time (over " << seams_to_remove << " iterations): " << total_compute_seam_time << " ms\n";
    std::cout << "  Total processing time: " << total_time << " ms\n";
    std::cout << "OK wrote " << outF << "\n";

    return 0;
}

