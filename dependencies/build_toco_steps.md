## Tensorflow r1.13

git clone https://github.com/tensorflow/tensorflow.git

cd tensorflow

git fetch origin r1.13:r1.13

git checkout r1.13

## Install Openjdk-8 (on Ubuntu)
sudo apt-get install openjdk-8-jdk

## Bazel (on Ubuntu)
https://github.com/bazelbuild/bazel/releases/download/0.19.2/bazel_0.19.2-linux-x86_64.deb

sudo dpkg -i bazel_0.19.2-linux-x86_64.deb

## 更新pip源
pip install pip -U

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

## build toco
bazel build -c opt tensorflow/lite/toco:toco

## add to path
mv bazel-bin <PLACE_FOR_DEPENDENCIES>

export PATH="<PLACE_FOR_DEPENDENCIES>/bazel-bin/tensorflow/lite/toco":$PATH

