from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="Sentry_finModel_1.pt", data="datasets/Deer-Segmentation-8/data.yaml", imgsz=480, half=False, device="cpu")

