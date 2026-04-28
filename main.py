#!/usr/bin/env python3
"""tauv-pipe: SAM3 maskesi ile boru takip sistemi."""

import signal
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml
from cyclonedds.domain import DomainParticipant

from pipe_algorithm import MaskProcessor, PipeController, ProcessResult
from dds_interface import CommandPublisher, MaskSubscriber


def load_config(path: str = "config.yaml") -> dict:
    with open(Path(__file__).parent / path) as f:
        return yaml.safe_load(f)


def set_sam3_prompt(api_url: str, prompt: str):
    try:
        resp = requests.post(
            f"{api_url}/api/prompt",
            json={"camera": "bottom", "prompt": prompt},
            timeout=5,
        )
        if resp.ok:
            print(f"[SAM3] prompt = '{prompt}'")
        else:
            print(f"[SAM3] prompt ayarlanamadi: {resp.status_code}")
    except requests.RequestException as e:
        print(f"[SAM3] baglanti hatasi: {e}")


def main():
    cfg = load_config()
    print("=== tauv-pipe: Boru Takip Baslatiliyor ===")
    print(f"  forward_pwm    : {cfg.get('forward_pwm', 200)}")
    print(f"  max_yaw_pwm    : {cfg.get('max_yaw_pwm', 200)}")
    print(f"  kp_yaw         : {cfg.get('kp_yaw', 150.0)}")
    print(f"  ki_yaw         : {cfg.get('ki_yaw', 200.0)}")
    print(f"  num_slices     : {cfg.get('num_slices', 8)}")
    print(f"  ema_alpha      : {cfg.get('ema_alpha', 0.6)}")
    print()

    participant = DomainParticipant(domain_id=cfg.get("dds_domain", 0))

    from logger import init_dds_logging
    init_dds_logging(participant, "tauv_pipe")

    mask_sub = MaskSubscriber(participant, cfg.get("mask_topic", "sam3/bottom/segmentation_mask"))
    cmd_pub = CommandPublisher(participant, cfg.get("command_topic", "embedded/control/stream_command"))

    processor = MaskProcessor(
        num_slices=cfg.get("num_slices", 8),
        slice_weights=cfg.get("slice_weights"),
        min_mask_area=cfg.get("min_mask_area", 500),
    )

    controller = PipeController(
        neutral_pwm=cfg.get("neutral_pwm", 1500),
        forward_pwm=cfg.get("forward_pwm", 200),
        max_yaw_pwm=cfg.get("max_yaw_pwm", 200),
        kp_yaw=cfg.get("kp_yaw", 150.0),
        ki_yaw=cfg.get("ki_yaw", 200.0),
        ema_alpha=cfg.get("ema_alpha", 0.6),
        turn_curvature_thresh=cfg.get("turn_curvature_thresh", 0.03),
        turn_align_thresh_deg=cfg.get("turn_align_thresh_deg", 10.0),
        turn_forward_pwm=cfg.get("turn_forward_pwm", 150),
        turn_yaw_boost=cfg.get("turn_yaw_boost", 3.0),
        coast_timeout=cfg.get("coast_timeout", 1.5),
        coast_yaw_decay=cfg.get("coast_yaw_decay", 0.55),
        reverse_yaw_pwm=cfg.get("reverse_yaw_pwm", 150),
        reacquire_forward_pwm=cfg.get("reacquire_forward_pwm", 100),
        reacquire_duration_s=cfg.get("reacquire_duration_s", 5.0),
    )

    shutdown = False

    def on_signal(sig, frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    set_sam3_prompt(cfg.get("sam3_api_url", "http://localhost:5003"), cfg.get("sam3_prompt", "pipe"))

    mask_sub.start()
    print("[OK] Maske dinleniyor, kontrol dongusu basladi.\n")

    timeout = cfg.get("mask_timeout_sec", 3.0)
    mask_threshold = cfg.get("mask_threshold", 128)
    no_mask_count = 0

    try:
        while not shutdown:
            mask = mask_sub.get_mask(timeout=timeout)

            if mask is None:
                no_mask_count += 1
                cmd = controller.compute(None)
                cmd_pub.send(cmd)
                ts = time.strftime("%H:%M:%S")
                coasting = cmd["yaw"] != cfg.get("neutral_pwm", 1500)
                status = "COAST" if coasting else "DUR"
                print(f"[{ts}] MASKE YOK (x{no_mask_count}) -> {status}  yaw={cmd['yaw']}")
                continue

            no_mask_count = 0

            raw = np.frombuffer(bytes(mask.mask_data), dtype=np.uint8).reshape(mask.height, mask.width)
            _, binary = cv2.threshold(raw, mask_threshold, 255, cv2.THRESH_BINARY)

            result = processor.process(binary)
            cmd = controller.compute(result)
            cmd_pub.send(cmd)

            ts = time.strftime("%H:%M:%S")
            if result.error is not None:
                state = controller.state
                cont_str = "DEVAM" if result.pipe_continues else "SON"
                turn_str = ""
                if result.turn_direction != 0:
                    d = "SAG" if result.turn_direction > 0 else "SOL"
                    turn_str = f"  DONUS={d}({result.curvature:+.2f})"
                print(
                    f"[{ts}] [{state}] error={result.error:+.3f}  "
                    f"angle={result.pipe_angle_deg:+.1f}  "
                    f"yaw={cmd['yaw']}  fwd={cmd['forward']}  "
                    f"slices={len(result.slice_centroids)}/{cfg.get('num_slices', 8)}  "
                    f"scan={result.scan_hit_count}({cont_str})"
                    f"{turn_str}"
                )
            else:
                print(f"[{ts}] pipe algilanmadi (area={result.total_area}) -> DUR")

    finally:
        print("\n[SHUTDOWN] Neutral komut gonderiliyor...")
        cmd_pub.send(controller.stop_cmd())
        mask_sub.stop()
        print("[SHUTDOWN] Tamam.")


if __name__ == "__main__":
    main()
