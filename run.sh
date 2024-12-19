 torchrun --nproc_per_node 2 train_for_inpaint.py --train_dataset "6000 @ BlendedMVS(ROOT='data/blendedmvs_processed', split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 3 * Co3d(split='train', ROOT='data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" --test_dataset "1000 @ BlendedMVS(ROOT='data/blendedmvs_processed', split='val', resolution=(512,384), seed=777) + 100 @ Co3d(split='test', ROOT='data/co3d_subset_processed', resolution=(512,384), seed=777)" --model "DUSt3R_InpaintModel(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" --pretrained "pretrained/DUSt3R_ViTLarge_BaseDecoder_512_dpt/model.safetensors" --lr 0.0001 --min_lr 1e-06 --warmup_epochs 15 --epochs 150 --batch_size 4 --accum_iter 2 --save_freq 10 --keep_freq 5 --eval_freq 1 --disable_cudnn_benchmark --output_dir "checkpoints/dust3r_inpaint_in_blendedmvs" > blend.log