import torch, torch.nn.functional as F, time
def run(label, x, w):
    F.conv3d(x, w); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t = time.perf_counter(); y = F.conv3d(x, w); torch.cuda.synchronize(); dtm = time.perf_counter() - t
    extra = (torch.cuda.max_memory_allocated() - base - y.numel()*y.element_size())/2**20
    print(f"{label:40s} extra={extra:7.0f} MiB {dtm*1000:7.1f} ms", flush=True)
bf = torch.bfloat16
x = torch.randn(1,96,3,34,34,device="cuda",dtype=bf); w=torch.randn(96,96,3,3,3,device="cuda",dtype=bf)
run("tiny bf16", x, w)
x = torch.randn(1,96,6,482,834,device="cuda",dtype=bf)
run("full bf16 contiguous", x, w)
run("full bf16 channels_last_3d", x.to(memory_format=torch.channels_last_3d), w.to(memory_format=torch.channels_last_3d))
torch.backends.cudnn.benchmark = True
run("full bf16 cudnn.benchmark", x, w)
torch.backends.cudnn.benchmark = False
xs = x.unsqueeze(0)  # N=1 still
run("full bf16, split T into 2 calls (approx)", x[:, :, :4], w)
