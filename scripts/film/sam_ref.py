"""Reference masks for the carry test: SAM 2.1 (tiny, off the shelf) on every frame of the shot, from clicks on frame 0."""
import sys, subprocess, time
import numpy as np, torch
from transformers import Sam2VideoModel, Sam2VideoProcessor
import json
SRC, W, H = "data/tears_of_steel_720p.mov", 640, 268; START, T = float(sys.argv[2]), int(sys.argv[3]); clicks = {int(k): v for k, v in json.loads(sys.argv[4]).items()}; K = len(clicks)
p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", str(START), "-i", SRC, "-frames:v", str(T), "-vf", f"scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, check=True)
col = np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
dev = "cuda"; name = "facebook/sam2.1-hiera-tiny"
model = Sam2VideoModel.from_pretrained(name).to(dev).eval(); proc = Sam2VideoProcessor.from_pretrained(name)
sess = proc.init_video_session(video=[f for f in col], inference_device=dev, processing_device="cpu", video_storage_device="cpu", max_vision_features_cache_size=2)
proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=list(clicks), input_points=[[clicks[k] for k in clicks]], input_labels=[[[1] * len(clicks[k]) for k in clicks]])
masks = np.zeros((len(col), K, H, W), bool); t0 = time.time()
with torch.inference_mode():
    out = model(inference_session=sess, frame_idx=0)
    for o in model.propagate_in_video_iterator(sess):
        m = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=False)[0]
        masks[o.frame_idx] = (m.float().cpu().numpy().reshape(-1, H, W) > 0)[:K]
dt = time.time() - t0
print(f"SAM 2.1 tiny: {len(col)} frames in {dt:.1f} s, {1000 * dt / len(col):.0f} ms a frame; GPU peak {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB")
print("mask share of frame, first / middle / last:", [[round(float(masks[t, k].mean()), 3) for t in (0, len(col) // 2, len(col) - 1)] for k in range(K)])
np.savez_compressed(sys.argv[1], masks=masks, col=col)
