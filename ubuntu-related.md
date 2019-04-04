# Ubuntu related commands 

## File ownership & right change
* After running docker container as root, need to change folder & file's ownership and rights
```bash
$ sudo chown -R moono:moono /path/to/dir
$ sudo chmod -R 755 /path/to/dir
```

## NVIDIA CUDA version check
```bash
# CUDA
$ cat /usr/local/cuda/version.txt

# cuDNN
$ cat /usr/include/cudnn.h | grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL"
```
