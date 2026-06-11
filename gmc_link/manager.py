"""
Inference Manager linking tracks to text prompts using physical vectors and spatial models.

UPDATED: Now uses cumulative homography method for better numerical stability
and debugging capabilities.
"""
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import torch
import numpy as np

from .utils import (
    normalize_velocity,
    MotionBuffer,
    ScoreBuffer,
    warp_points,
    VELOCITY_SCALE,
)
from .alignment import MotionLanguageAligner
from .core import ORBHomographyEngine
from .dataset import compute_per_track_extras
from .ego.ego_router import EgoRouter, make_ego_router


class GMCLinkManager:
    """
    GMC-Link Manager: Orchestrates Motion Estimation and Reasoning modules.

    Uses cumulative homography method:
    - Stores ORIGINAL centroid coordinates (never modified)
    - Stores CUMULATIVE homographies: H[t] transforms frame[t] -> current_frame
    - Warps coordinates ONCE when computing velocities (not every frame)
    
    This provides better numerical stability and debugging capabilities.
    """

    # Multi-scale frame gaps matching training
    FRAME_GAPS = [2, 5, 10]  # short, mid, long

    # World-XY projection scale: tunes 95th-percentile world dX/dY input to ~1.0
    # so MLP inputs sit near unit magnitude after Z/f projection. Calibrated
    # against KITTI car-speed distribution (~0.5 m/frame at 10 fps).
    VELOCITY_SCALE_WORLD = 2.0

    def __init__(
        self,
        weights_path: str = None,
        device: str = "cpu",
        lang_dim: int = 384,
        frame_gap: int = 10,  # max gap for buffer sizing
        ego_router: "EgoRouter | str | None" = None,
        use_depth: bool = False,
        world_xy: bool = False,
    ) -> None:
        self.device = device
        self.frame_gap = frame_gap

        self.motion_buffer = MotionBuffer(alpha=0.3)
        self.score_buffer = ScoreBuffer(alpha=0.4)
        self.cosine_buffer = ScoreBuffer(alpha=0.4)

        self.extra_features: list = []
        motion_dim = 13
        self.temperature = 1.0
        # CLIP-feat (Exp 39): ckpt meta drives aligner ctor + runtime extraction.
        self.use_clip_feat = False
        self.clip_feat_dim = 512
        self.clip_proj_dim = 64
        self.fusion_site = "input_concat"
        self.lang_passthrough = False
        self.app_proj_dim = 256
        self.architecture = "mlp"
        self.seq_len = 10  # temporal_transformer window T (ckpt overrides)
        self.depth_ego = "cohort"  # Path B: "oxts" when ckpt trained --depth-source lidar_oxts
        checkpoint = None
        if weights_path:
            checkpoint = torch.load(weights_path, map_location=device)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                motion_dim = checkpoint.get("motion_dim", 13)
                self.extra_features = checkpoint.get("extra_features") or []
                self.temperature = checkpoint.get("temperature", 1.0)
                ckpt_lang_dim = checkpoint.get("lang_dim")
                if ckpt_lang_dim is not None:
                    lang_dim = ckpt_lang_dim
                ckpt_use_depth = checkpoint.get("use_depth")
                if ckpt_use_depth is not None:
                    use_depth = bool(ckpt_use_depth)
                ckpt_world_xy = checkpoint.get("world_xy")
                if ckpt_world_xy is not None:
                    world_xy = bool(ckpt_world_xy)
                self.use_clip_feat = bool(checkpoint.get("use_clip_feat", False))
                self.clip_feat_dim = int(checkpoint.get("clip_feat_dim") or 512)
                self.clip_proj_dim = int(checkpoint.get("clip_proj_dim") or 64)
                self.fusion_site = str(checkpoint.get("fusion_site") or "input_concat")
                self.lang_passthrough = bool(checkpoint.get("lang_passthrough", False))
                self.app_proj_dim = int(checkpoint.get("app_proj_dim") or 256)
                self.architecture = str(checkpoint.get("architecture") or "mlp")
                ckpt_seq_len = checkpoint.get("seq_len")
                if ckpt_seq_len:
                    self.seq_len = int(ckpt_seq_len)
                if checkpoint.get("depth_source") == "lidar_oxts":
                    self.depth_ego = "oxts"  # Path B: GT oxts ego in the 17D depth path

        # 17D depth path
        self.use_depth = bool(use_depth)
        if self.use_depth and motion_dim == 13:
            motion_dim = 17
        self.motion_dim = motion_dim

        # World-XY projection: swap image dx/dy for metric world dX/dY
        # via inverse pinhole `dX = dx_pixel * Z / f_x`. Same dim, drop-in.
        self.world_xy = bool(world_xy)
        if self.world_xy:
            from gmc_link.camera_intrinsics import CameraIntrinsics
            self.intrinsics = CameraIntrinsics()
        else:
            self.intrinsics = None
        # per-track Z history: {track_id: [(frame_id, z_meters), ...]}
        self._z_history: Dict[int, list] = {}

        self.aligner = MotionLanguageAligner(
            motion_dim=motion_dim, lang_dim=lang_dim, embed_dim=256,
            architecture=self.architecture,
            seq_len=self.seq_len,
            use_clip_feat=self.use_clip_feat,
            clip_feat_dim=self.clip_feat_dim,
            clip_proj_dim=self.clip_proj_dim,
            fusion_site=self.fusion_site,
            lang_passthrough=self.lang_passthrough,
            app_proj_dim=self.app_proj_dim,
        ).to(device)
        # Lazy CLIP B/32 (DataComp-XL) for runtime bbox-crop encoding. Tracker
        # bboxes are NOT in the GT-keyed train cache, must extract live.
        self._clip_model = None
        self._clip_preprocess = None
        if self.use_clip_feat:
            import open_clip
            from PIL import Image as _PILImage
            self._PILImage = _PILImage
            m, _, pp = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="datacomp_xl_s13b_b90k"
            )
            self._clip_model = m.to(device).eval()
            for p in self._clip_model.parameters():
                p.requires_grad = False
            self._clip_preprocess = pp
        if checkpoint is not None:
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                self.aligner.load_state_dict(checkpoint["model"])
            else:
                self.aligner.load_state_dict(checkpoint)
        self.aligner.eval()

        if ego_router is None:
            self.ego_engine = ORBHomographyEngine(max_features=1500)
        elif isinstance(ego_router, str):
            self.ego_engine = make_ego_router(ego_router)
        else:
            self.ego_engine = ego_router
        self.prev_frame = None
        self.prev_detections = None

        # CUMULATIVE HOMOGRAPHY: Store original coordinates (never warped)
        self.centroid_history: Dict[int, deque] = {}
        self.wh_history: Dict[int, deque] = {}
        # temporal_transformer: per-track window of the last seq_len per-frame
        # motion vectors (the same vectors the MLP path consumes one at a time)
        self.sequence_history: Dict[int, deque] = {}
        # window_stats extra: per-track trailing (dx_m, dy_m) steps (seqB2)
        self.wstat_history: Dict[int, deque] = {}
        # Per-track scale-velocity history for accel_multiscale: {tid: [(frame_idx, scale_vels)]}
        # frame_idx is the manager's live frame counter (monotone); deque keeps last max_gap+1 entries.
        self.scale_vel_history: Dict[int, deque] = {}
        self._frame_counter: int = -1

        # Store cumulative homographies: H[i] warps frame[t-i] -> current frame
        self.homography_buffer: deque = deque(maxlen=frame_gap + 1)

        # Background residual buffer for noise floor estimation
        self.bg_residual_buffer: deque = deque(maxlen=frame_gap + 1)

    def encode_clip_image_bboxes(
        self,
        frame: np.ndarray,
        bboxes_xyxy: List[Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """Run CLIP B/32 image encoder on bbox crops. Returns (N, clip_feat_dim) fp32.
        Used by both process_frame (1-pass) and FH V2 builder (2-pass)."""
        if not self.use_clip_feat or self._clip_model is None:
            raise RuntimeError("encode_clip_image_bboxes requires use_clip_feat=True ckpt")
        if not bboxes_xyxy:
            return torch.empty(0, self.clip_feat_dim, device=self.device)
        import cv2 as _cv2
        img_h, img_w = frame.shape[:2]
        crops = []
        for (bx1, by1, bx2, by2) in bboxes_xyxy:
            bx1i = int(max(0, min(img_w - 1, bx1)))
            by1i = int(max(0, min(img_h - 1, by1)))
            bx2i = int(max(bx1i + 1, min(img_w, bx2)))
            by2i = int(max(by1i + 1, min(img_h, by2)))
            crop_bgr = frame[by1i:by2i, bx1i:bx2i]
            if crop_bgr.size == 0:
                crop_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
            crop_rgb = _cv2.cvtColor(crop_bgr, _cv2.COLOR_BGR2RGB)
            pil = self._PILImage.fromarray(crop_rgb)
            crops.append(self._clip_preprocess(pil))
        crop_batch = torch.stack(crops, dim=0).to(self.device)
        with torch.no_grad():
            feats = self._clip_model.encode_image(crop_batch)
        return feats.float()

    def _compute_dz_residual(
        self,
        z_now: Dict[int, float],
        z_prev: Dict[int, float],
        stationary_ids: set,
    ) -> Dict[int, float]:
        """Per-track dZ with ego-Z compensation.

        dZ_ego = median(dZ) over stationary tracks; dZ_residual = dZ_track − dZ_ego.
        Falls back to zero compensation if no stationary tracks.
        """
        dz_raw = {tid: z_now[tid] - z_prev[tid] for tid in z_now if tid in z_prev}
        stat_dz = [dz_raw[tid] for tid in stationary_ids if tid in dz_raw]
        if stat_dz:
            dz_ego = float(np.median(stat_dz))
        else:
            dz_ego = 0.0
        return {tid: v - dz_ego for tid, v in dz_raw.items()}

    def process_frame(
        self,
        frame: np.ndarray,
        active_tracks: List[Any],
        language_embedding: torch.Tensor,
        detections: Optional[np.ndarray] = None,
        update_state: bool = True,
        raw_cos: bool = False,
        depth_z_lookup: Optional[Dict[int, float]] = None,
        seq: Optional[str] = None,
        frame_id: Optional[int] = None,
        clip_feat_cache: Optional[dict] = None,
    ) -> Tuple[Dict[int, float], Dict[int, np.ndarray], Dict[int, float]]:
        """
        Process a frame: compute centroid-difference velocities per tracked object,
        and return alignment scores against a language prompt.

        Args:
            frame: (H, W, 3) The current video frame.
            active_tracks: List of track objects (must have `id` and `centroid`).
            language_embedding: (1, L_dim) Tensor representing the language prompt.
            detections: (N, 4) array of bounding boxes for ego-motion masking.
            update_state: Whether to update internal state (for multiple evaluations per frame)
            raw_cos: If True, scores_dict returns raw cosine in [-1, +1] (no sigmoid, no EMA).
                     For Arm B fusion experiments where downstream uses raw cosine directly.

        Returns:
            scores_dict: {track_id: alignment score} — if raw_cos=False (default), sigmoid+EMA
                         smoothed [0,1]; if raw_cos=True, raw cosine [-1,+1] (no smoothing).
            velocities_dict: {track_id: 13D motion vector}
            cosine_dict: {track_id: EMA-smoothed cosine similarity} (raw cosine, asymmetric EMA)
        """
        if not active_tracks:
            return {}, {}, {}

        if update_state:
            self._frame_counter += 1

        img_h, img_w = frame.shape[:2]
        frame_shape = (img_h, img_w)

        # CUMULATIVE HOMOGRAPHY UPDATE
        if update_state:
            if self.prev_frame is not None:
                # Estimate H_{t-1 -> t} and background warp residual
                H_prev_to_curr, bg_residual = self.ego_engine.estimate_homography(
                    self.prev_frame, frame, self.prev_detections
                )
                self.bg_residual_buffer.append(bg_residual)

                # Update ALL cumulative homographies by composing with new homography
                updated_homographies = deque(maxlen=self.frame_gap + 1)
                for H_old in self.homography_buffer:
                    # H_old maps frame[t_old] -> frame[t-1]
                    # H_prev_to_curr maps frame[t-1] -> frame[t]
                    # Composition: frame[t_old] -> frame[t]
                    H_cumulative = H_prev_to_curr @ H_old
                    updated_homographies.append(H_cumulative)
                
                # Current frame has identity homography (maps to itself)
                updated_homographies.append(np.eye(3, dtype=np.float32))
                self.homography_buffer = updated_homographies
            else:
                # First frame: identity homography
                self.homography_buffer.append(np.eye(3, dtype=np.float32))

            self.prev_frame = frame.copy()
            if detections is not None:
                self.prev_detections = [tuple(d) for d in detections]
            else:
                self.prev_detections = None

        # Pre-compute background noise floor (shared across all tracks)
        if len(self.bg_residual_buffer) > 0:
            bg_stack = np.array(list(self.bg_residual_buffer))
            bg_max = np.max(np.abs(bg_stack), axis=0)
            bg_magnitude = np.sqrt(
                (bg_max[0] / float(img_w) * VELOCITY_SCALE) ** 2
                + (bg_max[1] / float(img_h) * VELOCITY_SCALE) ** 2
            )
        else:
            bg_magnitude = 0.0

        track_ids = []
        compensated_velocities = []
        # CLIP path: collect bbox crops per appended track for batch encode after loop.
        clip_bbox_xyxy: List[Tuple[int, int, int, int]] = []
        # Depth path: collect per-track raw dZ-per-gap, defer ego-Z comp until after loop
        per_track_z_now: Dict[int, float] = {}
        per_track_dz_raw: Dict[int, Dict[int, float]] = {}
        per_track_res_speed: Dict[int, float] = {}

        for track in active_tracks:
            if not hasattr(track, "centroid") or track.centroid is None:
                continue

            tid = track.id
            curr_centroid = np.array(track.centroid, dtype=np.float64)

            if tid not in self.centroid_history:
                self.centroid_history[tid] = deque(maxlen=self.frame_gap + 1)
                self.wh_history[tid] = deque(maxlen=self.frame_gap + 1)

            # Store ORIGINAL coordinates (never warp!)
            if update_state:
                self.centroid_history[tid].append(curr_centroid)

            if hasattr(track, "bbox") and track.bbox is not None:
                bx1, by1, bx2, by2 = track.bbox
                curr_w = bx2 - bx1
                curr_h = by2 - by1
            else:
                curr_w, curr_h = 0.0, 0.0

            if update_state:
                self.wh_history[tid].append(np.array([curr_w, curr_h], dtype=np.float64))

            # Get original coordinate history
            centroid_hist = list(self.centroid_history[tid])
            wh_hist = list(self.wh_history[tid])

            if len(centroid_hist) > 1:
                # PolarMOT-inspired ego-invariant decomposition
                T = len(centroid_hist)
                homographies = list(self.homography_buffer)[-T:]
                while len(homographies) < T:
                    homographies.insert(0, np.eye(3, dtype=np.float32))

                # Multi-scale: residual velocity = raw - ego
                residual_velocities = []
                for gap in self.FRAME_GAPS:
                    if T > gap:
                        # Raw velocity (normalized)
                        v_raw = centroid_hist[-1] - centroid_hist[-(gap + 1)]
                        v_norm = normalize_velocity(v_raw, frame_shape)
                        # Per-object ego displacement (normalized)
                        old_c = centroid_hist[-(gap + 1)]
                        H_old = homographies[T - 1 - gap]
                        warped_old = warp_points(np.array([old_c]), H_old)[0]
                        ego_v = normalize_velocity(warped_old - old_c, frame_shape)
                        # Residual = raw - ego (object-only motion)
                        residual_velocities.append(v_norm - ego_v)
                    else:
                        residual_velocities.append(np.zeros(2, dtype=np.float32))

                # dw, dh from mid-scale (index 1, gap=5) or full history
                mid_gap = self.FRAME_GAPS[1]
                if T > mid_gap:
                    raw_dw_dh = wh_hist[-1] - wh_hist[-(mid_gap + 1)]
                else:
                    raw_dw_dh = wh_hist[-1] - wh_hist[0]
                dw_raw = raw_dw_dh[0] / float(img_w) * VELOCITY_SCALE
                dh_raw = raw_dw_dh[1] / float(img_h) * VELOCITY_SCALE

                # Smooth the 8D residual velocity + dw/dh
                full_raw_v = np.array([
                    residual_velocities[0][0], residual_velocities[0][1],
                    residual_velocities[1][0], residual_velocities[1][1],
                    residual_velocities[2][0], residual_velocities[2][1],
                    dw_raw, dh_raw,
                ], dtype=np.float32)
                if update_state:
                    smoothed_v = self.motion_buffer.smooth(tid, full_raw_v)
                else:
                    smoothed_v = self.motion_buffer.peek(tid, full_raw_v)
                dx_s, dy_s = smoothed_v[0], smoothed_v[1]
                dx_m, dy_m = smoothed_v[2], smoothed_v[3]
                dx_l, dy_l = smoothed_v[4], smoothed_v[5]
                dw, dh = smoothed_v[6], smoothed_v[7]
            else:
                # First appearance: zero velocity
                smoothed_v = np.zeros(8, dtype=np.float32)
                dx_s, dy_s = 0.0, 0.0
                dx_m, dy_m = 0.0, 0.0
                dx_l, dy_l = 0.0, 0.0
                dw, dh = 0.0, 0.0

            # Build 13D vector: residual velocity + spatial
            w_n = curr_w / float(img_w)
            h_n = curr_h / float(img_h)
            cx_n = curr_centroid[0] / float(img_w)
            cy_n = curr_centroid[1] / float(img_h)

            # SNR from mid-scale residual speed (bg_magnitude pre-computed above)
            obj_speed = np.sqrt(dx_m ** 2 + dy_m ** 2)
            snr = obj_speed / (bg_magnitude + 1e-6)

            spatial_motion = np.array(
                [dx_s, dy_s, dx_m, dy_m, dx_l, dy_l,
                 dw, dh, cx_n, cy_n, w_n, h_n, snr], dtype=np.float32
            )

            # World-XY projection: swap image dx/dy for metric world dX/dY.
            # Each smoothed slot s_norm = (pixel_dx / W) * VELOCITY_SCALE.
            # Recover pixels (× W / VELOCITY_SCALE), project (× Z / f), re-normalize
            # (× VELOCITY_SCALE_WORLD). Combined factor = (W/VELOCITY_SCALE)·(Z/f)·SCALE_WORLD.
            # snr stays image-domain (slot 12) — bg + obj ratio coherent only in pixel space.
            if self.world_xy and self.intrinsics is not None and seq is not None:
                f_x, f_y, _, _ = self.intrinsics.get(seq)
                z_raw = None
                if depth_z_lookup is not None:
                    z_raw = depth_z_lookup.get(tid)
                z_eff = float(z_raw) if z_raw is not None else 30.0
                z_eff = max(1.0, min(80.0, z_eff))
                sx = (img_w / VELOCITY_SCALE) * z_eff / f_x * self.VELOCITY_SCALE_WORLD
                sy = (img_h / VELOCITY_SCALE) * z_eff / f_y * self.VELOCITY_SCALE_WORLD
                spatial_motion[0] *= sx  # dx_s
                spatial_motion[1] *= sy  # dy_s
                spatial_motion[2] *= sx  # dx_m
                spatial_motion[3] *= sy  # dy_m
                spatial_motion[4] *= sx  # dx_l
                spatial_motion[5] *= sy  # dy_l

            if self.extra_features:
                # Per-track (non-relational) extras only — manager has no neighbor context.
                per_track_names = [
                    f for f in self.extra_features
                    if f in {"speed_m", "heading_m", "accel", "ego_motion",
                             "accel_multiscale", "heading_sincos", "window_stats"}
                ]
                if per_track_names:
                    scale_velocities = [(dx_s, dy_s), (dx_m, dy_m), (dx_l, dy_l)]
                    window_vel_hist = None
                    if "window_stats" in per_track_names:
                        # Mirror dataset.py: trailing <=10 mid-scale steps incl.
                        # current. Peek mode (update_state=False) must not mutate.
                        wh = self.wstat_history.setdefault(tid, deque(maxlen=10))
                        if update_state:
                            wh.append((dx_m, dy_m))
                            window_vel_hist = list(wh)
                        else:
                            window_vel_hist = (list(wh) + [(dx_m, dy_m)])[-10:]
                    accel_per_scale = None
                    if "accel_multiscale" in per_track_names:
                        if tid not in self.scale_vel_history:
                            self.scale_vel_history[tid] = deque(maxlen=self.frame_gap + 1)
                        hist = self.scale_vel_history[tid]
                        counter = self._frame_counter
                        accel_per_scale = []
                        for gap_idx, gap in enumerate(self.FRAME_GAPS):
                            past = None
                            for (pc, pv) in reversed(hist):
                                lag = counter - pc
                                if lag >= gap and lag <= 2 * gap:
                                    past = pv
                                    break
                            if past is None:
                                accel_per_scale.append((0.0, 0.0))
                            else:
                                now_dx, now_dy = scale_velocities[gap_idx]
                                past_dx, past_dy = past[gap_idx]
                                accel_per_scale.append((
                                    (now_dx - past_dx) / gap,
                                    (now_dy - past_dy) / gap,
                                ))
                        if update_state:
                            hist.append((counter, list(scale_velocities)))
                    extras = compute_per_track_extras(
                        per_track_names, scale_velocities,
                        accel_per_scale=accel_per_scale,
                        window_vel_hist=window_vel_hist,
                    )
                    spatial_motion = np.concatenate(
                        [spatial_motion, np.array(extras, dtype=np.float32)]
                    )

            # Depth path: collect raw dZ per gap & current Z (ego comp deferred)
            if self.use_depth and depth_z_lookup is not None:
                z_now = depth_z_lookup.get(tid)
                per_track_z_now[tid] = z_now
                per_track_res_speed[tid] = float(np.sqrt(dx_m ** 2 + dy_m ** 2))
                if z_now is not None:
                    hist = self._z_history.get(tid, [])
                    z_at_gap: Dict[int, Optional[float]] = {g: None for g in self.FRAME_GAPS}
                    for past_fid, past_z in reversed(hist):
                        lag = self._frame_counter - past_fid
                        for g in self.FRAME_GAPS:
                            if z_at_gap[g] is None and lag >= g:
                                z_at_gap[g] = past_z
                    dz_raw_g: Dict[int, float] = {}
                    for g in self.FRAME_GAPS:
                        if z_at_gap[g] is not None:
                            dz_raw_g[g] = z_now - z_at_gap[g]
                    per_track_dz_raw[tid] = dz_raw_g
                    if update_state:
                        self._z_history.setdefault(tid, []).append((self._frame_counter, z_now))
                        max_gap = max(self.FRAME_GAPS)
                        self._z_history[tid] = [
                            (f, z) for f, z in self._z_history[tid]
                            if self._frame_counter - f <= max_gap + 1
                        ]
                else:
                    per_track_dz_raw[tid] = {}

            track_ids.append(tid)
            compensated_velocities.append(spatial_motion)
            if self.use_clip_feat:
                if hasattr(track, "bbox") and track.bbox is not None:
                    bx1, by1, bx2, by2 = track.bbox
                    bx1i = int(max(0, min(img_w - 1, round(bx1))))
                    by1i = int(max(0, min(img_h - 1, round(by1))))
                    bx2i = int(max(bx1i + 1, min(img_w, round(bx2))))
                    by2i = int(max(by1i + 1, min(img_h, round(by2))))
                    clip_bbox_xyxy.append((bx1i, by1i, bx2i, by2i))
                else:
                    clip_bbox_xyxy.append((0, 0, 1, 1))

        # Post-loop: ego-Z compensation per gap (cohort median over stationary tracks)
        # Stationary criterion: residual mid-scale speed < 0.01 (≈1 px/frame at KITTI W=1242).
        if self.use_depth and depth_z_lookup is not None:
            dz_ego_per_gap: Dict[int, float] = {}
            if self.depth_ego == "oxts":
                # Path B: GT oxts ego-ΔZ per gap, backward interval [now-g, now] to
                # match dz_raw = z_now - z_past. frame_id is the 1-based cache key ->
                # 0-based pose idx = frame_id-1. Nominal gap g (one value per gap,
                # mirroring the cohort path). Frame alignment proven in
                # tests/test_pathB_oxts_frame_alignment.py.
                from gmc_link import kitti_tracking_gt as K
                poses, calib = K.seq_poses_calib(seq)
                now_img = (frame_id - 1) if frame_id is not None else 0
                for g in self.FRAME_GAPS:
                    dz_ego_per_gap[g] = K.ego_dz_camera(poses, calib, now_img, g)
            else:
                stat_ids = {tid for tid, mag in per_track_res_speed.items() if mag < 0.01}
                for g in self.FRAME_GAPS:
                    stat_dz = [per_track_dz_raw[tid][g]
                               for tid in stat_ids
                               if tid in per_track_dz_raw and g in per_track_dz_raw[tid]]
                    dz_ego_per_gap[g] = float(np.median(stat_dz)) if stat_dz else 0.0
            for i, tid in enumerate(track_ids):
                z_now = per_track_z_now.get(tid)
                if z_now is None:
                    depth_4d = np.zeros(4, dtype=np.float32)
                else:
                    z_n = float(np.clip(z_now, 0.0, 80.0) / 100.0)
                    dz_g = per_track_dz_raw.get(tid, {})
                    dz_residual = []
                    for g in self.FRAME_GAPS:
                        if g in dz_g:
                            dz_residual.append((dz_g[g] - dz_ego_per_gap[g]) / 10.0)
                        else:
                            dz_residual.append(0.0)
                    depth_4d = np.array([z_n, *dz_residual], dtype=np.float32)
                compensated_velocities[i] = np.concatenate(
                    [compensated_velocities[i], depth_4d]
                ).astype(np.float32)

        if not compensated_velocities:
            return {}, {}, {}

        # Align motion with language via cosine similarity
        padding_mask = None
        if self.architecture == "temporal_transformer":
            # Window the per-frame vectors exactly like training's
            # _vectors_to_sequences: last <=T frames incl. current, FRONT-padded
            # with zeros, mask True at positions 1..n_padded (position 0 = CLS).
            # Emits from the track's first vector on (never silent) — a 1-frame
            # window is a valid training sample too.
            T = self.seq_len
            dim = compensated_velocities[0].shape[0]
            seq_batch = np.zeros((len(track_ids), T, dim), dtype=np.float32)
            mask_batch = np.zeros((len(track_ids), T + 1), dtype=bool)
            for i, tid in enumerate(track_ids):
                hist = self.sequence_history.setdefault(tid, deque(maxlen=T))
                if update_state:
                    hist.append(compensated_velocities[i])
                    window = list(hist)
                else:
                    window = (list(hist) + [compensated_velocities[i]])[-T:]
                n_valid = len(window)
                seq_batch[i, T - n_valid:] = np.stack(window)
                mask_batch[i, 1 : T - n_valid + 1] = True
            motion_tensor = torch.tensor(seq_batch, dtype=torch.float32).to(self.device)
            padding_mask = torch.tensor(mask_batch).to(self.device)
        else:
            motion_tensor = torch.tensor(
                np.array(compensated_velocities), dtype=torch.float32
            ).to(self.device)

        # Runtime CLIP B/32 forward on tracker bbox crops (Exp 39 path).
        # CLIP features are expression-independent (same frame+boxes across exprs),
        # so cache per frame_id when a shared cache is supplied — avoids re-encoding
        # the identical crops once per expression.
        clip_tensor = None
        if self.use_clip_feat and self._clip_model is not None:
            if clip_feat_cache is not None and frame_id is not None:
                clip_tensor = clip_feat_cache.get(frame_id)
                if clip_tensor is None:
                    clip_tensor = self.encode_clip_image_bboxes(frame, clip_bbox_xyxy)
                    clip_feat_cache[frame_id] = clip_tensor
            else:
                clip_tensor = self.encode_clip_image_bboxes(frame, clip_bbox_xyxy)

        with torch.no_grad():
            motion_emb, lang_emb = self.aligner.encode(
                motion_tensor, language_embedding.to(self.device),
                padding_mask=padding_mask,
                clip_feats=clip_tensor,
            )
            # Case 2 fusion-transformer spike: stash per-track pre-cosine
            # embeddings so callers can dump them without re-running aligner.
            # Keyed by track_id; overwritten each frame.
            self._last_motion_emb = {
                int(tid): motion_emb[i].detach().cpu().numpy()
                for i, tid in enumerate(track_ids)
            }
            self._last_lang_emb = lang_emb.detach().cpu().numpy()
            # Cosine similarity with margin calibration → sigmoid to [0, 1]
            # Margin shifts the sigmoid reference point so that zero-similarity
            # maps below 0.5, improving discrimination for stationary objects
            cosine_sim = torch.matmul(motion_emb, lang_emb.t()).flatten()
            cosine_sim_np = cosine_sim.cpu().numpy()
            margin = 0.05  # calibrated from GT/non-GT cosine distributions
            raw_scores = torch.sigmoid((cosine_sim - margin) / self.temperature).cpu().numpy()

        # Apply score smoothing for temporal consistency
        scores_dict = {}
        velocities_dict = {}
        cosine_dict = {}
        for i, tid in enumerate(track_ids):
            raw_score = float(raw_scores[i])
            raw_cosine = float(cosine_sim_np[i])
            if update_state:
                smoothed_score = self.score_buffer.smooth(tid, raw_score)
                smoothed_cosine = self.cosine_buffer.smooth(tid, raw_cosine)
            else:
                smoothed_score = self.score_buffer.peek(tid, raw_score)
                smoothed_cosine = self.cosine_buffer.peek(tid, raw_cosine)
            scores_dict[tid] = raw_cosine if raw_cos else smoothed_score
            velocities_dict[tid] = compensated_velocities[i]
            cosine_dict[tid] = smoothed_cosine

        if update_state:
            # Clean up dead tracks
            active_ids = set(track_ids)
            self.motion_buffer.clear_dead_tracks(track_ids)
            self.score_buffer.clear_dead_tracks(track_ids)
            self.cosine_buffer.clear_dead_tracks(track_ids)
            dead_centroids = set(self.centroid_history.keys()) - active_ids
            for d in dead_centroids:
                del self.centroid_history[d]
                if d in self.wh_history:
                    del self.wh_history[d]
                if d in self.scale_vel_history:
                    del self.scale_vel_history[d]
                if d in self.sequence_history:
                    del self.sequence_history[d]
                if d in self.wstat_history:
                    del self.wstat_history[d]

        return scores_dict, velocities_dict, cosine_dict
