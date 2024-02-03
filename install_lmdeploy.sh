#!/bin/bash

# 安装lmdeploy
# 获取安装lmdeploy的位置下的lib文件夹路径
lmdeploy_dir=$(pip show lmdeploy | grep Location | cut -d' ' -f2)
lib_dir="${lmdeploy_dir}/lmdeploy/lib"

# 检查lib目录是否存在
if [ ! -d "$lib_dir" ]
then
    echo "Lib directory does not exist at ${lib_dir}"
    exit 1
fi

# 克隆lmdeploy的仓库
git clone https://github.com/InternLM/lmdeploy.git || exit 1

# 将lib文件夹拷贝到刚刚克隆的lmdeploy下
cp -r "$lib_dir" "lmdeploy/lmdeploy/" || exit 1

pip uninstall -y lmdeploy

cd lmdeploy && git checkout v0.2.1 && cd ..
mv lmdeploy lmdeploy-backup
mv lmdeploy-backup/lmdeploy lmdeploy

echo "Script executed successfully"