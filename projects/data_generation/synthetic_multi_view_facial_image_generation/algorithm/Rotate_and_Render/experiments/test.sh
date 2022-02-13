python3  -u test_frontal.py  \
        --names rs_ijba3 \
        --dataset megafaceprobe \
        --list_start 0 \
        --list_end 5000 \
        --dataset_mode single \
        --gpu_ids 0,1,2,3,4,5,6,7 \
        --netG rotatespade \
        --norm_G spectralsyncbatch \
        --batchSize 18 \
        --model rotatespade \
        --label_nc 5 \
        --nThreads 3 \
        --heatmap_size 2.5 \
        --chunk_size 40 40\
        --no_gaussian_landmark \
        --multi_gpu \
        --device_count 8\
        --render_thread 6 \
        --label_mask \
        --align \
        #--use_BG \
        #--chunk_size 2 4 4 4 4 4\
                
