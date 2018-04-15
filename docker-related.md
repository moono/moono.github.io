# Docker related useful links

## Using docker with smb (cifs) share
* Test share folder in local first
```bash
# create mount point first via 'mkdir'
$ sudo mount -t cifs -o username=USER_NAME,password='PASSWORD' //IP_ADDRESS/PATH /tmp/tempshare

# if it says about 'operation not supported' ...
# do following
$ modprobe cifs
$ echo 7 > /proc/fs/cifs/cifsFYI
# bla bla blah 
$ dmesg

# find something like ...
# CIFS VFS: Dialect not supported by server. Consider specifying
# vers=1.0 or vers=2.1 on mount for accessing older servers

# now add options and mount again
$ sudo mount -t cifs -o username=USER_NAME,password='PASSWORD',vers=1.0 //IP_ADDRESS/PATH /tmp/tempshare
```
* Now create docker volume
```bash
# create docker volume
$ docker volume create --driver local \
    --opt type=cifs \
    --opt device=//IP_ADDRESS/PATH \
    --opt o=username=USER_NAME,password='PASSWORD',vers=1.0 \
    --name YOUR_VOLUME_NAME

# check docker volume
$ docker volume inspect YOUR_VOLUME_NAME
# ...
# ...

# run with volume
$ docker run --rm -it -v YOUR_VOLUME_NAME:/path/to/inside/container IMAGE_NAME /bin/bash

# check folders inside container...
root@abcdefg:/# cd /path/to/inside/container
root@abcdefg:/path/to/inside/container# ls -al
# ...
# ...
```