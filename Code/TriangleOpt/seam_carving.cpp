// seam_carving.cpp

#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>

//------------------------------------------------------------------------------
// Custom image and matrix classes
//------------------------------------------------------------------------------

class Img {
public:
    std::vector<uint32_t> pixels;
    int width, height, stride;

    Img(int w, int h)
        : pixels(w*h), width(w), height(h), stride(w) {}

    inline uint32_t& at(int r, int c) {
        assert(r>=0 && r<height && c>=0 && c<width);
        return pixels[r*stride + c];
    }
    inline const uint32_t& at(int r, int c) const {
        assert(r>=0 && r<height && c>=0 && c<width);
        return pixels[r*stride + c];
    }
};

class MatrixF {
public:
    std::vector<float> items;
    int width, height, stride;

    MatrixF(int w, int h)
        : items(w*h), width(w), height(h), stride(w) {}

    inline float& at(int r, int c) {
        assert(r>=0 && r<height && c>=0 && c<width);
        return items[r*stride + c];
    }
    inline const float& at(int r, int c) const {
        assert(r>=0 && r<height && c>=0 && c<width);
        return items[r*stride + c];
    }
};

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

static float rgb_to_lum(uint32_t bgra) {
    float b = ((bgra >>  0) & 0xFF)/255.f;
    float g = ((bgra >>  8) & 0xFF)/255.f;
    float r = ((bgra >> 16) & 0xFF)/255.f;
    return 0.2126f*r + 0.7152f*g + 0.0722f*b;
}

static void luminance(const Img &img, MatrixF &lum) {
    for(int y=0;y<img.height;++y)
      for(int x=0;x<img.width;++x)
        lum.at(y,x)=rgb_to_lum(img.at(y,x));
}

static float sobel_filter_at(const MatrixF &m, int cx, int cy) {
    static float gx[3][3]={{1,0,-1},{2,0,-2},{1,0,-1}};
    static float gy[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};
    float sx=0, sy=0;
    for(int dy=-1;dy<=1;++dy)
      for(int dx=-1;dx<=1;++dx) {
        int x=cx+dx, y=cy+dy;
        float v=(y>=0&&y<m.height&&x>=0&&x<m.width)?m.at(y,x):0;
        sx+=v*gx[dy+1][dx+1];
        sy+=v*gy[dy+1][dx+1];
      }
    return sx*sx+sy*sy;
}

static void sobel_filter(const MatrixF &in, MatrixF &out) {
    for(int y=0;y<in.height;++y)
      for(int x=0;x<in.width;++x)
        out.at(y,x)=sobel_filter_at(in,x,y);
}

//------------------------------------------------------------------------------
// Two-phase triangular DP with chain-width batching
//------------------------------------------------------------------------------

static void grad_to_dp(const MatrixF &grad, MatrixF &dp,
                       int num_strips, int chain_width)
{
    int H = grad.height, W = grad.width;
    int strip_h = (H + num_strips - 1)/num_strips;

    // first row
    #pragma omp parallel for
    for(int x=0;x<W;++x) dp.at(0,x)=grad.at(0,x);

    for(int s=0;s<num_strips;++s){
      int y0 = 1 + s*strip_h;
      int y1 = std::min(H, y0+strip_h);
      if(y0>=y1) break;
      int h = y1-y0;

      // Phase 1: downward triangles in batches of 'chain_width' diagonals
      #pragma omp parallel for schedule(dynamic)
      for(int t=0; t< W+h-1; t+=chain_width){
        for(int p=0; p<chain_width; ++p){
          int tt = t+p;
          if(tt>=W+h-1) break;
          for(int i=0;i<h;++i){
            int x = tt - i;
            if(x<0||x>=W) continue;
            int y = y0 + i;
            float m = dp.at(y-1,x);
            if(x>0)    m = std::min(m, dp.at(y-1,x-1));
            if(x+1<W) m = std::min(m, dp.at(y-1,x+1));
            dp.at(y,x)= grad.at(y,x) + m;
          }
        }
      }

      // Phase 2: upward triangles in batches
      #pragma omp parallel for schedule(dynamic)
      for(int t = 1 - W; t < h; t += chain_width){
        for(int p=0;p<chain_width;++p){
          int tt = t+p;
          if(tt>=h) break;
          for(int i=0;i<h;++i){
            int x = tt + i;
            if(x<0||x>=W) continue;
            int y = y0 + (h-1) - i;
            float m = dp.at(y-1,x);
            if(x>0)    m = std::min(m, dp.at(y-1,x-1));
            if(x+1<W) m = std::min(m, dp.at(y-1,x+1));
            dp.at(y,x)= grad.at(y,x) + m;
          }
        }
      }
    }
}

//------------------------------------------------------------------------------
// Rest of DP & seam functions (unchanged)
//------------------------------------------------------------------------------

static void img_remove_column_at_row(Img &img,int r,int c){
  auto*p=img.pixels.data()+r*img.stride;
  std::move_backward(p+c+1,p+img.width,p+img.width);
}
static void mat_remove_column_at_row(MatrixF &m,int r,int c){
  auto*p=m.items.data()+r*m.stride;
  std::move_backward(p+c+1,p+m.width,p+m.width);
}

static void compute_seam(const MatrixF &dp,std::vector<int>&s){
  int H=dp.height, W=dp.width;
  s.resize(H);
  s[H-1]=std::min_element(&dp.at(H-1,0),&dp.at(H-1,0)+W) - &dp.at(H-1,0);
  for(int y=H-2;y>=0;--y){
    int px=s[y+1]; float best=dp.at(y,px);
    s[y]=px;
    for(int d=-1;d<=1;++d){
      int x=px+d;
      if(x>=0&&x<W&&dp.at(y,x)<best){
        best=dp.at(y,x);
        s[y]=x;
      }
    }
  }
}

static void markout_sobel_patches(MatrixF &g,const std::vector<int>&s){
  for(int y=0;y<g.height;++y){
    int x=s[y];
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
      int ny=y+dy, nx=x+dx;
      if(ny>=0&&ny<g.height&&nx>=0&&nx<g.width)
        reinterpret_cast<uint32_t&>(g.at(ny,nx))=0xFFFFFFFF;
    }
  }
}

//------------------------------------------------------------------------------
// Main + timing + CLI parsing
//------------------------------------------------------------------------------

static void print_usage(const char* pr){
  std::cout<<"Usage: "<<pr
    <<" [-proc <T>] [-strips <S>] [-width <W>] in.png out.png\n";
}

int main(int argc,char*argv[]){
  int argi=1;
  int num_threads = omp_get_max_threads();
  int num_strips  = 4;
  int chain_width = 1;

  if(argc>1 && std::string(argv[argi])=="-proc"){
    num_threads=std::atoi(argv[argi+1]);
    omp_set_num_threads(num_threads);
    argi+=2;
  }
  if(argc>argi && std::string(argv[argi])=="-strips"){
    num_strips=std::atoi(argv[argi+1]);
    argi+=2;
  }
  if(argc>argi && std::string(argv[argi])=="-width"){
    chain_width=std::atoi(argv[argi+1]);
    if(chain_width<1) chain_width=1;
    argi+=2;
  }
  if(argc-argi<2){
    print_usage(argv[0]);
    return 1;
  }

  std::string inF=argv[argi++], outF=argv[argi++];
  std::cout<<"Threads="<<num_threads
           <<"  Strips="<<num_strips
           <<"  ChainWidth="<<chain_width<<"\n";

  cv::Mat in = cv::imread(inF,cv::IMREAD_UNCHANGED);
  if(in.empty()){std::cerr<<"ERROR loading\n";return 1;}
  cv::Mat bgra;
  if(in.channels()==3) cv::cvtColor(in,bgra,cv::COLOR_BGR2BGRA);
  else if(in.channels()==4) bgra=in.clone();
  else{std::cerr<<"Bad channels\n";return 1;}

  int W=bgra.cols, H=bgra.rows;
  Img img(W,H);
  if(bgra.isContinuous()){
    std::memcpy(img.pixels.data(),bgra.data,W*H*4);
  } else {
    for(int y=0;y<H;++y)
      std::memcpy(img.pixels.data()+y*img.stride,
                  bgra.ptr(y),W*4);
  }

  MatrixF lum(W,H), grad(W,H), dp(W,H);
  std::vector<int> seam(H);
  int seams_to_remove = W/4;

  double t_lum=0, t_sobel=0, t_dp=0, t_seam=0;
  auto T0 = std::chrono::high_resolution_clock::now();

  { auto A = std::chrono::high_resolution_clock::now();
    luminance(img,lum);
    auto B = std::chrono::high_resolution_clock::now();
    t_lum = std::chrono::duration<double,std::milli>(B-A).count();
  }
  { auto A = std::chrono::high_resolution_clock::now();
    sobel_filter(lum,grad);
    auto B = std::chrono::high_resolution_clock::now();
    t_sobel = std::chrono::duration<double,std::milli>(B-A).count();
  }

  for(int i=0;i<seams_to_remove;++i){
    auto A = std::chrono::high_resolution_clock::now();
    grad_to_dp(grad,dp,num_strips,chain_width);
    auto B = std::chrono::high_resolution_clock::now();
    t_dp += std::chrono::duration<double,std::milli>(B-A).count();

    auto C = std::chrono::high_resolution_clock::now();
    compute_seam(dp,seam);
    auto D = std::chrono::high_resolution_clock::now();
    t_seam += std::chrono::duration<double,std::milli>(D-C).count();

    markout_sobel_patches(grad,seam);
    for(int y=0;y<H;++y){
      int x=seam[y];
      img_remove_column_at_row(img,y,x);
      mat_remove_column_at_row(lum,y,x);
      mat_remove_column_at_row(grad,y,x);
    }
    --W; --lum.width; --grad.width; --dp.width;

    // update around seam
    for(int y=0;y<H;++y){
      for(int x=seam[y];x<W;++x)
        if(reinterpret_cast<uint32_t&>(grad.at(y,x))==0xFFFFFFFF)
          grad.at(y,x)=sobel_filter_at(lum,x,y);
      for(int x=seam[y]-1;x>=0;--x){
        if(reinterpret_cast<uint32_t&>(grad.at(y,x))==0xFFFFFFFF)
          grad.at(y,x)=sobel_filter_at(lum,x,y);
        else break;
      }
    }
  }

  auto T1 = std::chrono::high_resolution_clock::now();
  double t_total = std::chrono::duration<double,std::milli>(T1-T0).count();

  cv::Mat out(H,W,CV_8UC4,img.pixels.data(),img.stride*4);
  cv::imwrite(outF,out);

  std::cout<<"\nSummary Timing (ms):\n"
           <<"  luminance:    "<<t_lum  <<"\n"
           <<"  Sobel filter: "<<t_sobel<<"\n"
           <<"  grad_to_dp:   "<<t_dp   <<"\n"
           <<"  compute_seam: "<<t_seam <<"\n"
           <<"  Total:        "<<t_total<<"\n";

  return 0;
}
