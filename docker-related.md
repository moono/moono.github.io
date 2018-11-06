# Docker related useful links

## Find specific string in the logs of an docker container
* ref: https://github.com/moby/moby/issues/9508
```bash
$ docker log <container_id_or_name> 2>&1 | grep <word_to_find>
```

## run docker container with specific user name & group
* ref: https://medium.com/@mccode/understanding-how-uid-and-gid-work-in-docker-containers-c37a01d01cf
```bash
# check your user id & group id
$ id
# will output ... 
# uid=1001(moono) gid=1001(moono) groups=1001(moono),27(sudo),999(docker)

$ docker run --user=1001:1001 -it <image_name> /bin/bash
```

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

## Delete 'None' tag images
```bash
$ docker image prune
```
