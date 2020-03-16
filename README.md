# Jing-asv

写给静静的ASV组件  

## 安装  

注意： 工具包有使用pytorch和numpy，建议使用conda新建环境后安装。  

1. 安装工具包  
   从GitHub下载包  

   ```bash
   git clone https://github.com/LampV/kiwi
   ```

   进入文件夹

   ```bash
   cd kiwi
   ```

   安装wjwgym到本地

   ```bash
   pip install ./jasv
    ```

2. 运行示例  

   ```bash
   python asv_main.py
   ```

   若程序正常运行，说明安装成功

## 使用  

1. 通过gym创建环境

   ```python
   import gym
   import jasv
   # 使用gym方式获取env
   env = gym.make('JASV-v0')
   ```

2. 直接创建env

   ```python
   from jasv import ASVEnv
   env = ASVEnv()
   ```
