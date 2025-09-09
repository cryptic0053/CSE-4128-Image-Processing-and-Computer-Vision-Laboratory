Solar Panel Defect Detection (Tuned Classical CV Pipeline)
Key outputs:
  00_input.png                 - original grayscale input
  00b_hist_matched.png         - (if provided) histogram-matched to reference
  01_hist_eq.png               - CLAHE local contrast enhancement
  02_gaussian_denoised.png     - Gaussian denoised
  02b_crack_score.png          - inverted + black-hat crack score
  04_canny_edges.png           - Canny edges (on notched image if enabled)
  05_crack_mask_raw.png        - initial crack mask before refinement
  05_crack_mask.png            - refined crack mask (blob removal / thin-elongated kept)
  07_fft_highpass.png          - FFT high-pass visualization
  07b_notch_filtered.png       - (if enabled) notch-filtered image
  08_regions_overlay.png       - labeled regions overlay
  09_skeleton.png              - skeleton used for crack length
  10_hough_lines_filtered.png  - grid-suppressed Hough lines
  metrics.json / regions_metrics.csv - quantitative descriptors
