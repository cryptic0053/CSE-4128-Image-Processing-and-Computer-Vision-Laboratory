Solar Panel Defect Detection — v4 (Frangi + Multiscale BH)
Key outputs:
  00_input.png
  00b_hist_matched.png         - if provided
  01_hist_eq.png
  02_gaussian_denoised.png
  02b_crack_score.png          - blend of invert + multiscale black-hat + frangi
  04_canny_edges_raw.png       - before grid suppression
  04_canny_edges.png           - after grid suppression (if enabled)
  05_crack_mask_raw.png        - mask before refinement
  05_crack_mask.png            - refined mask (curvy, elongated)
  07_fft_highpass.png
  07b_notch_filtered.png       - if notch enabled
  08_regions_overlay.png
  09_skeleton.png              - pruned skeleton used for crack length
  10_hough_lines_filtered.png
  metrics.json / regions_metrics.csv
