# ICVL-Gaussian Denoising
python test_icvl_gaussian.py --arch=sst_icvl --device='cuda' --index=1 --ckpt=model_zoo/sst/checkpoints_gaussian.pth --save_dir=./results/sst

python test_icvl_gaussian.py --arch=sert_icvl --device='cuda' --index=1 --ckpt=model_zoo/sert/icvl_gaussian.pth --save_dir=./results/sert

python test_icvl_gaussian.py --arch=grnet_icvl --device='cuda' --index=1 --ckpt=model_zoo/grnet/grn_gaussian.pth --save_dir=./results/grnet

python test_icvl_gaussian.py --arch=hsdt_icvl --device='cuda' --index=1 --ckpt=model_zoo/hsdt/hsdt_l_gaussian.pth --save_dir=./results/hsdt

python test_icvl_gaussian.py --arch=qrnn3d_icvl --device='cuda' --index=1 --ckpt=model_zoo/qrnn3d/qrnn3d_gaussian.pth --save_dir=./results/qrnn3d

python test_icvl_gaussian.py --arch=macnet --device='cuda' --index=1 --ckpt=model_zoo/macnet/macnet_gaussian.pth --save_dir=./results/macnet

python test_icvl_gaussian.py --arch=t3sc_icvl --device='cuda' --index=0 --ckpt=model_zoo/t3sc/t3sc_gaussian.pth --save_dir=./results/t3sc

python test_icvl_gaussian.py --arch=proposed_base_icvl --device='cuda' --index=1 --ckpt=model_zoo/proposed_dw/icvl_gaussian_base.ckpt --save_dir=./results/proposed_base

python test_icvl_gaussian.py --arch=proposed_tiny_icvl --device='cuda' --index=1 --ckpt=model_zoo/proposed_dw/icvl_gaussian_tiny.ckpt --save_dir=./results/proposed_tiny

# ICVL-Complex Denoising

python test_icvl_complex.py --arch=sst_icvl --device='cuda' --index=1 --ckpt=model_zoo/sst/checkpoints_complex.pth --save_dir=./results/sst

python test_icvl_complex.py  --arch=sert_icvl --device='cuda' --index=1 --ckpt=model_zoo/sert/icvl_complex.pth --save_dir=./results/sert

python test_icvl_complex.py  --arch=grnet_icvl --device='cuda' --index=0 --ckpt=model_zoo/grnet/grn_complex.pth --save_dir=./results/grnet

python test_icvl_complex.py  --arch=hsdt_icvl --device='cuda' --index=0 --ckpt=model_zoo/hsdt/hsdt_l_complex.pth --save_dir=./results/hsdt

python test_icvl_complex.py  --arch=qrnn3d_icvl --device='cuda' --index=0 --ckpt=model_zoo/qrnn3d/qrnn3d_complex.pth --save_dir=./results/qrnn3d

python test_icvl_complex.py  --arch=macnet --device='cuda' --index=1 --ckpt=model_zoo/macnet/macnet_complex.pth --save_dir=./results/macnet

python test_icvl_complex.py  --arch=t3sc_icvl --device='cuda' --index=0 --ckpt=model_zoo/t3sc/t3sc_complex.pth --save_dir=./results/t3sc

python test_icvl_complex.py  --arch=proposed_base_icvl --device='cuda' --index=0 --ckpt=model_zoo/proposed_dw/icvl_complex_base.ckpt --save_dir=./results/proposed_base

python test_icvl_complex.py  --arch=proposed_tiny_icvl --device='cuda' --index=0 --ckpt=model_zoo/proposed_dw/icvl_complex_tiny.ckpt --save_dir=./results/proposed_tiny

# Realistic Denoising

python test_realistic.py --arch=sst_real --device='cuda' --index=0 --ckpt=model_zoo/sst/sst_realistic.pth --save_dir=./results/sst

python test_realistic.py  --arch=sert_real --device='cuda' --index=1 --ckpt=model_zoo/sert/real_realistic.pth --save_dir=./results/sert

python test_realistic.py  --arch=grnet_real --device='cuda' --index=0 --ckpt=model_zoo/grnet/GRNET_real_net.pth --save_dir=./results/grnet

python test_realistic.py  --arch=qrnn3d_real --device='cuda' --index=0 --ckpt=model_zoo/qrnn3d/qrnn3d_real_net.pth --save_dir=./results/qrnn3d

python test_realistic.py  --arch=t3sc_real --device='cuda' --index=0 --ckpt=model_zoo/t3sc/t3sc_real_net.pth --save_dir=./results/t3sc

python test_realistic.py  --arch=macnet --device='cuda' --index=1 --ckpt=model_zoo/macnet/macnet_real_net.pth --save_dir=./results/macnet

python test_realistic.py  --arch=proposed_base_real --device='cuda' --index=0 --ckpt=model_zoo/proposed_dw/realistic_base.ckpt --save_dir=./results/proposed_base



python test_icvl_gaussian.py --arch=uformer --device='cuda' --index=1 --ckpt=model_zoo/uformer/uformer_gaussian.ckpt --save_dir=./results/uformer

python test_icvl_gaussian.py --arch=scunet --device='cuda' --index=1 --ckpt=model_zoo/scunet/scunet_gaussian.ckpt --save_dir=./results/scunet