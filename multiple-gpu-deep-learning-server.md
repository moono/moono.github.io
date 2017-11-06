# How to setup deep learning server for beginners

## Target readers
* Wants to build deep learning server with multiple gpus
* Not so good at linux commands
* Not so familiar with docker
* Got tired of searching for answers to setup deep learning environment
* Just want to run deep neural network
* Done some coding with single gpu and wants to use multiple gpus now

## In a Nutshell
* Buy a computer with multiple gpus(probably Nvidia GTX 1080 Ti or Titan X or higher)
* Setup docker environment
* Connect to docker environment server Pycharm professional(remote debugging)
* Run specific (i.e. gpu isolation) gpus that I want

## Before start
* As I'm a too beginer in setting these kind of environment, please point me to right direction if the following contents are wrong

## Server system
* GPU: Titan X - 4 ea

## Setting the server

### Prerequisites
* This [guide][link-build-dl-system] helps build/setup simple deep learning computer.
* Install Ubuntu 14.04 or 16.04 LTS
* Setup software update download location to near you
  * System Settings -> Software & Updates -> Ubuntu Software -> Download from
* Install software updates
```bash
$ sudo apt-get update
$ sudo apt-get upgrade
```
* Install graphics driver (There are many tutorials out there but I just enabled the driver like below)
  * System Settings -> Software & Updates -> Additional Drivers -> Select 'Using NVIDIA binary driver - version xxx.xx .....'
  * I'm using the version 384.90
  * **_reboot_**
  * My failure
    * I used **_display port_** cable at first and when ever I install the graphics driver with different kind of methods in the internet, all failed to show up the login screen. 
    * Screen outputs the error something like 'the system is running in low-graphics mode'
    * I finally managed to solve this problem by switching the cable to **_dvi_**. What an odd behaviour...
  * check if following command works
    * `$ nvidia-smi`
    * it will print something like following...
    
    ![][img-host-nvidia-smi]
    
* (No need for docker users) _(optional)_ Additional nvidia related software 
  * cuda 8.0
  * cudnn 5.1 or 6.0 related to cuda 8.0 (need to sign up for nvidia to download)
* _(optional)_ Setup the user accounts for users who will use this server

### Docker

#### Install docker
* Follow the steps in [Install docker - Ubuntu][link-docker-install]
* _(optional)_ Try docker tutorial at least [Part1 & Part2][link-docker-tutorial] to know what docker is

#### Install nvidia-docker
* Follow this [guide][link-nvidia-docker-install]
* After installing nvidia-docker, test with `nvidia-smi` inside that docker container
```bash
# Test nvidia-smi
# option 1. use latest tag to see if it works
$ nvidia-docker run --rm nvidia/cuda nvidia-smi

# option 2. use specific cuda version tag if option 1 is giving you a error...
$ nvidia-docker run --rm nvidia/cuda:8.0 nvidia-smi

# Explanation
# `nvidia-docker`: using nvidia-docker rather than docker to use fully functional nvidia gpu related support
# `run`: start (run) conatiner
# `--rm`: remove container after work is done
# `nvidia/cuda:<tag>`: docker image ro run --> if doesn't exists than it will pull from https://hub.docker.com/ (similar to github)
# `nvidia-smi`: command to run when container starts
```

#### Download (pull) appropriate docker image & test 
* Personally I prefer [floyhub docker images][link-floyhub-docker-images]
  * example: [tensorflow 1.3.0 with gpu on python3.x](https://github.com/floydhub/dockerfiles/blob/master/dl/tensorflow/1.3.0/Dockerfile-py3.gpu_aws)
    ```bash
    # pull image
    $ docker pull floydhub/tensorflow:1.3.0-gpu-py3_aws.12
      
    # run with nvidia-docker, map port 8888 to 8888 because this image already installed jupyter notebook inside like(tensorflow/tensorflow)
    $ nvidia-docker run -it -p 8888:8888 floydhub/tensorflow:1.3.0-gpu-py3_aws.12
    ```
* try to connect to jupyter notebook as seen in terminal screen
* run something to check if working: i.e. `sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))`

#### Stack ssh into existing docker image so that we can access to it - [link][link-docker-ssh]
* create empty folder anywhere you want and `cd` into that
* create a file named 'Dockerfile'
* type below in 'Dockerfile' and save
```bash
# base image to start with
FROM floydhub/tensorflow:1.3.0-gpu-py3_aws.12
  
# install ssh related package & basic setup
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
    
# change the password (here is 'docker') to yours
RUN echo 'root:docker' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
    
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
    
# I have no idea what these does
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
    
# ssh port is 22 so let it be exposed
EXPOSE 22
    
# run below command when container starts
CMD ["/usr/sbin/sshd", "-D"]
```
* Now build, run & test the 'Dockerfile'
```bash
# build the above 'Dockerfile' as named 'my_awesome_tf'
$ docker build -t my_awesome_tf .
    
# run built docker image 'my_awesome_tf' with nvidia-docker
$ nvidia-docker run -d -P --name test_tf my_awesome_tf
    
# find out which port 22 is mapped to in real world
$ docker port test_tf 22
0.0.0.0:40234

# connect to container with username 'root' via ssh (password is docker)
$ ssh root@localhost -p 40234
    
root@f38c87f2a42d:/#
```

#### Link with Pycharm (setup remote access - must be pycharm professional)
  * Follow this [link](https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d#3830) to setup pycharm remote
  * use password auth type instead


## Etc

### Helpful docker commands - from [here](https://docs.docker.com/get-started/part2/#recap-and-cheat-sheet-optional)
| command | meaning |
| ------- | ------- |
| docker run <options> | run docker with given options |
| docker run -it <...> | interactive mode |
| docker run -p <exposed port>:<container port> | map ports |
| docker run -v <local directory>:<container directory>	| map directory |
| docker build | -t friendlyname . | Create image using this directory's Dockerfile |
| docker run -p 4000:80 friendlyname | Run "friendlyname" mapping port 4000 to 80 |
| docker run -d -p 4000:80 friendlyname | Same thing, but in detached mode |
| docker container ls | List all running containers |
| docker container ls -a | List all containers, even those not running |
| docker container stop <hash> | Gracefully stop the specified container |
| docker container kill <hash> | Force shutdown of the specified container |
| docker container rm <hash> | Remove specified container from this machine |
| docker container rm $(docker container ls -a -q) | Remove all containers |
| docker image ls -a | List all images on this machine |
| docker image rm <image id> | Remove specified image from this machine |
| docker image rm $(docker image ls -a -q) | Remove all images from this machine |
| docker login | Log in this CLI session using your Docker credentials |
| docker tag <image> username/repository:tag | Tag <image> for upload to registry |
| docker push username/repository:tag | Upload tagged image to registry |
| docker run username/repository:tag | Run image from a registry |

### Common network port numbers - from [here](http://www.utilizewindows.com/list-of-common-network-port-numbers/)
| Port | name | Transport protocol |
| ---- | ---- | ------------------ |
| 20, 21 | File Transfer Protocol (FTP) | TCP |
| 22 | Secure Shell (SSH) | TCP and UDP |
| 23 | Telnet | TCP |
| 25 | Simple Mail Transfer Protocol (SMTP) | TCP |
| 50, 51 | IPSec | - |
| 53 | Domain Name Server (DNS) | TCP and UDP |
| 67, 68 | Dynamic Host Configuration Protocol (DHCP) | UDP |
| 69 | Trivial File Transfer Protocol (TFTP) | UDP |
| 80 | HyperText Transfer Protocol (HTTP) | TCP |
| 110 | Post Office Protocol (POP3) | TCP |
| 119 | Network News Transport Protocol (NNTP) | TCP |
| 123 | Network Time Protocol (NTP) | UDP |
| 135-139 | NetBIOS | TCP and UDP |
| 143 | Internet Message Access Protocol (IMAP4) | TCP and UDP |
| 161, 162 | Simple Network Management Protocol (SNMP) | TCP and UDP |
| 389 | Lightweight Directory Access Protocol | TCP and UDP |
| 443 | HTTP with Secure Sockets Layer (SSL) | TCP and UDP |


# References

* https://docs.docker.com/get-started/
* https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/
* https://github.com/NVIDIA/nvidia-docker
* https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d
* https://github.com/floydhub/dockerfiles

[img-host-nvidia-smi]: ./assets/host-nvidia-smi.PNG
[link-build-dl-system]: https://medium.com/towards-data-science/build-and-setup-your-own-deep-learning-server-from-scratch-e771dacaa252
[link-docker-install]: https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/
[link-docker-tutorial]: https://docs.docker.com/get-started/
[link-nvidia-docker-install]: https://github.com/NVIDIA/nvidia-docker/wiki/Installation
[link-docker-hub]: https://hub.docker.com/
[link-floyhub-docker-images]: https://github.com/floydhub/dockerfiles
[link-docker-ssh]: https://docs.docker.com/engine/examples/running_ssh_service/

