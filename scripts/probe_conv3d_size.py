import torch, torch.nn.functional as F, time
def run(T, H, W, dt, C=96):
    x = torch.randn(1, C, T, H, W, device="cuda", dtype=dt); w = torch.randn(C, C, 3, 3, 3, device="cuda", dtype=dt)
    F.conv3d(x, w); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t = time.perf_counter(); y = F.conv3d(x, w); torch.cuda.synchronize(); dtm = time.perf_counter() - t
    extra = (torch.cuda.max_memory_allocated() - base - y.numel()*y.element_size())/2**20
    print(f"{str(dt)[6:]:9s} in={T}x{H}x{W} in_numel={x.numel()/1e6:.0f}M  extra_minus_out={extra:7.0f} MiB  {dtm*1000:7.1f} ms", flush=True)
    del x, w, y; torch.cuda.empty_cache()
for H, W in [(482, 834), (242, 834), (122, 834), (62, 834)]:
    run(6, H, W, torch.bfloat16)
run(6, 482, 834, torch.float32)
run(3, 482, 834, torch.bfloat16)
