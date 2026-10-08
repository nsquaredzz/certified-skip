// certskip CLI: reads raw 8-bit grey frames from stdin (e.g. an ffmpeg pipe)
// and prints per-frame keep statistics for the certified rule and, optionally,
// writes the keep masks as a raw uint8 stream.
//
//   ffmpeg -i cam.mp4 -vf fps=2,format=gray -f rawvideo - 2>/dev/null \
//     | ./certskip -w 1920 -h 1080 -p 16 -d 32 [--masks out.u8] [--quiet]
#include "certskip.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>

static void usage(const char* a) {
    std::fprintf(stderr,
        "usage: %s -w W -h H [-p PATCH=16] [-d DELTA=32] [--max-shift S] [--masks FILE] [--quiet]\n"
        "reads raw gray8 frames of size WxH from stdin\n", a);
}

int main(int argc, char** argv) {
    int W = 0, H = 0, P = 16; double delta = 32.0, max_shift = -1; const char* masks = nullptr; bool quiet = false;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* flag) { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", flag); std::exit(2); } return argv[++i]; };
        if (!std::strcmp(argv[i], "-w")) W = std::atoi(need("-w"));
        else if (!std::strcmp(argv[i], "-h")) H = std::atoi(need("-h"));
        else if (!std::strcmp(argv[i], "-p")) P = std::atoi(need("-p"));
        else if (!std::strcmp(argv[i], "-d")) delta = std::atof(need("-d"));
        else if (!std::strcmp(argv[i], "--max-shift")) max_shift = std::atof(need("--max-shift"));
        else if (!std::strcmp(argv[i], "--masks")) masks = need("--masks");
        else if (!std::strcmp(argv[i], "--quiet")) quiet = true;
        else { usage(argv[0]); return 2; }
    }
    if (W <= 0 || H <= 0 || P <= 0) { usage(argv[0]); return 2; }

    certskip::Params prm; prm.patch = P; prm.delta = delta; prm.max_shift = max_shift;
    certskip::Pruner<uint8_t> pr(H, W, prm);
    const size_t fsz = static_cast<size_t>(W) * H;
    const size_t n = static_cast<size_t>(pr.grid_h()) * pr.grid_w();
    std::vector<uint8_t> frame(fsz), keep(n);
    std::vector<float> spread(n), shift(n);
    FILE* mf = masks ? std::fopen(masks, "wb") : nullptr;
    if (masks && !mf) { std::perror("masks"); return 1; }

    long t = 0; double total_kept = 0; float worst_half_spread = 0;
    auto t0 = std::chrono::steady_clock::now();
    if (!quiet) std::printf("frame\tkept\tdrop_rate\tmax_half_spread\n");
    while (std::fread(frame.data(), 1, fsz, stdin) == fsz) {
        pr.step(frame.data(), keep.data(), spread.data(), shift.data());
        long kept = 0; float mhs = 0;
        for (size_t i = 0; i < n; ++i) { kept += keep[i]; if (!keep[i] && spread[i] * 0.5f > mhs) mhs = spread[i] * 0.5f; }
        if (mhs > worst_half_spread) worst_half_spread = mhs;
        total_kept += kept;
        if (!quiet) std::printf("%ld\t%ld\t%.4f\t%.1f\n", t, kept, 1.0 - double(kept) / n, mhs);
        if (mf) std::fwrite(keep.data(), 1, n, mf);
        ++t;
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    if (mf) std::fclose(mf);
    if (t == 0) { std::fprintf(stderr, "no frames read\n"); return 1; }
    std::fprintf(stderr,
        "frames=%ld grid=%dx%d patches/frame=%zu delta=%.1f\n"
        "overall drop rate=%.4f  certified max view error (grey levels)=%.1f\n"
        "throughput=%.1f frames/s (%.1f Mpx/s)\n",
        t, pr.grid_h(), pr.grid_w(), n, delta,
        1.0 - total_kept / (double(t) * n), worst_half_spread,
        t / secs, t * double(fsz) / 1e6 / secs);
    return 0;
}
