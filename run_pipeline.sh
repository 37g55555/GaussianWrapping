#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$DATASET_PATH/sparse" "$DATASET_PATH/dense"

colmap feature_extractor \
  --database_path "$DATASET_PATH/database.db" \
  --image_path "$DATASET_PATH/images"
colmap exhaustive_matcher \
  --database_path "$DATASET_PATH/database.db"
colmap mapper \
  --database_path "$DATASET_PATH/database.db" \
  --image_path "$DATASET_PATH/images" \
  --output_path "$DATASET_PATH/sparse"
colmap image_undistorter \
  --image_path "$DATASET_PATH/images" \
  --input_path "$DATASET_PATH/sparse/0" \
  --output_path "$DATASET_PATH/dense" \
  --output_type COLMAP

TRAIN_ARGS=(
  -s "$DATASET_PATH/dense" -m "outputs"
  --rasterizer ours
  --feature_dc_lr 0.0013
  --feature_rest_lr 0.00011
  --data_device cpu
  --N_max_gaussians 1500000
  --densify_until_iter 7000
  --test_iterations -1
)
python gaussian_wrapping/train.py "${TRAIN_ARGS[@]}" \
  -r 2 --iterations 7000 --checkpoint_iterations 7000
python gaussian_wrapping/train.py "${TRAIN_ARGS[@]}" \
  -r 4 --iterations 20000 --checkpoint_iterations 20000 \
  --start_checkpoint "outputs/chkpnt7000.pth"
python gaussian_wrapping/train.py "${TRAIN_ARGS[@]}" \
  -r 8 --start_checkpoint "outputs/chkpnt20000.pth"

python gaussian_wrapping/pivot_based_mesh_extraction.py \
  -s "$DATASET_PATH/dense" -m "outputs" -r 4 \
  --sdf_mode ours --isosurface_value 0.0 \
  --n_binary_steps 10 --use_valid_mask

python isolate_supported_object.py \
  -r "outputs/mesh_ours_2pivots.ply" \
  -s "$DATASET_PATH/dense" -m "outputs"
python export_fbx.py \
  -r "outputs/mesh_ours_2pivots_post.ply" \
  -s "$DATASET_PATH/dense" -m "outputs"
